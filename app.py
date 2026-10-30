"""
Main Flask Application for AI-Based Surface Defect Detection in Manufactured Components.
Provides RESTful APIs for camera capture ingestion, sequential CNN inference,
and SQLite inspection history management.
"""

import os
import sys
from flask import Flask, request, jsonify, send_from_directory

# Ensure project root is in python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.database import (
    init_db,
    insert_inspection,
    get_all_inspections,
    get_inspection_by_id,
    delete_inspection,
    clear_all_inspections,
    get_statistics,
)
from backend.image_processing import (
    decode_base64_image,
    save_image_to_disk,
    preprocess_image_for_model,
    assess_image_quality,
)
from backend.material_model import material_classifier
from backend.crack_model import crack_detector

# Flask Configuration
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "uploads")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Initialize database
init_db()

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")


# ---------------------------------------------------------------------------
# Static Frontend Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serves the primary industrial dashboard UI."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def serve_static(filename):
    """Serves static CSS, JS, and asset files."""
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    """Serves saved component inspection images."""
    return send_from_directory(UPLOADS_DIR, filename)


# ---------------------------------------------------------------------------
# REST API Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health_check():
    """Health and model availability telemetry."""
    material_ready = material_classifier.is_loaded
    crack_ready = crack_detector.is_loaded

    return jsonify({
        "status": "OK",
        "material_model_loaded": material_ready,
        "crack_model_loaded": crack_ready,
        "ai_models_loaded": material_ready and crack_ready,
        "material_error": material_classifier.error_message,
        "crack_error": crack_detector.error_message,
    })


@app.route("/api/inspect", methods=["POST"])
def inspect_component():
    """
    Primary Sequential Inspection Endpoint.
    1. Ingests component image (Base64 data URL or multipart file).
    2. Validates and saves original frame.
    3. Runs Material Identification CNN.
    4. If METAL -> Runs Surface Crack Detection CNN.
       If PLASTIC or OTHER -> Skips crack detection.
    5. Saves inspection record to SQLite.
    6. Returns structured inspection decision.
    """
    image_bytes = None
    filename_prefix = "capture"

    # Handle JSON base64 payload from webcam
    if request.is_json:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"success": False, "error": "No image payload provided in JSON request."}), 400
        try:
            image_bytes = decode_base64_image(data["image"])
        except Exception as e:
            return jsonify({"success": False, "error": f"Failed to decode base64 image: {str(e)}"}), 400

    # Handle multipart form upload fallback
    elif "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "error": "Empty file uploaded."}), 400
        image_bytes = file.read()
        filename_prefix = "upload"
    else:
        return jsonify({"success": False, "error": "Unsupported request format. Provide JSON or multipart file."}), 400

    if not image_bytes or len(image_bytes) < 100:
        return jsonify({"success": False, "error": "Image payload is empty or corrupted."}), 400

    # Save original component image to disk (Section 6: display original captured image)
    try:
        saved_filename, saved_filepath = save_image_to_disk(image_bytes, UPLOADS_DIR, prefix=filename_prefix)
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to save image: {str(e)}"}), 500

    # Image Quality / Lighting Assessment
    quality_check = assess_image_quality(image_bytes)

    # Preprocess image for CNN inference (Resizing, RGB conversion, [0, 1] normalization)
    try:
        preprocessed_batch = preprocess_image_for_model(saved_filepath, target_size=(224, 224))
    except Exception as e:
        return jsonify({"success": False, "error": f"Preprocessing failed: {str(e)}"}), 500

    # STEP 5: Material Classification
    material_result = material_classifier.predict(preprocessed_batch)
    detected_material = material_result.get("predicted_class", "OTHER / UNKNOWN")
    material_confidence = material_result.get("confidence", 0.0)

    # STEP 6: Sequential Logic Gating
    crack_result_data = {
        "predicted_class": "INSPECTION NOT PERFORMED",
        "surface_condition": "NOT EVALUATED",
        "confidence": 0.0,
        "is_performed": False,
        "note": ""
    }

    if detected_material == "METAL":
        # Metal identified -> Perform crack inspection
        crack_pred = crack_detector.predict(preprocessed_batch)
        crack_result_data = {
            "predicted_class": crack_pred.get("predicted_class", "NO VISIBLE CRACK DETECTED"),
            "surface_condition": crack_pred.get("surface_condition", "ACCEPTABLE"),
            "confidence": crack_pred.get("confidence", 0.0),
            "is_performed": True,
            "error": crack_pred.get("error"),
            "note": "Metal component analyzed for surface cracks."
        }
        status_flag = "COMPLETED"

    elif detected_material == "PLASTIC":
        # Plastic identified -> Skip crack detection
        crack_result_data["note"] = "Plastic component detected. Crack inspection is not performed."
        status_flag = "SKIPPED_PLASTIC"

    else:
        # Other / Unknown -> Skip crack detection
        detected_material = "OTHER / UNKNOWN"
        crack_result_data["note"] = "Material not recognised. Inspection cannot be performed."
        status_flag = "SKIPPED_UNKNOWN"

    # Store inspection result into SQLite database (Section 10)
    try:
        db_record = insert_inspection(
            image_filename=saved_filename,
            detected_material=detected_material,
            material_confidence=round(material_confidence * 100, 2),
            crack_result=crack_result_data["predicted_class"],
            crack_confidence=round(crack_result_data["confidence"] * 100, 2) if crack_result_data["is_performed"] else None,
            status=status_flag
        )
    except Exception as e:
        print(f"Database insertion error: {e}", file=sys.stderr)
        db_record = None

    return jsonify({
        "success": True,
        "material": material_result,
        "crack": crack_result_data,
        "quality": quality_check,
        "record": db_record,
        "image_url": f"/uploads/{saved_filename}"
    })


@app.route("/api/history", methods=["GET"])
def get_history():
    """Returns list of previous inspection records."""
    records = get_all_inspections()
    return jsonify(records)


@app.route("/api/history/<int:record_id>", methods=["DELETE"])
def delete_record(record_id):
    """Deletes an inspection record from SQLite."""
    success = delete_inspection(record_id)
    return jsonify({"success": success})


@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    """Clears all records from SQLite."""
    count = clear_all_inspections()
    return jsonify({"success": True, "deleted_count": count})


@app.route("/api/statistics", methods=["GET"])
def get_stats():
    """Aggregated summary statistics for the dashboard."""
    stats = get_statistics()
    return jsonify(stats)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("AI-BASED SURFACE DEFECT DETECTION IN MANUFACTURED COMPONENTS")
    print("=" * 65)
    print("Starting Flask local server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=True)

