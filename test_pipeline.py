import os, sys, base64
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from app import app

client = app.test_client()

print("1. HEALTH CHECK:")
res = client.get("/api/health")
health_data = res.get_json()
print("   Status:", health_data["status"])
print("   Material Model Loaded:", health_data["material_model_loaded"])
print("   Crack Model Loaded:", health_data["crack_model_loaded"])
print("   AI Engine Fully Ready:", health_data["ai_models_loaded"])
assert health_data["ai_models_loaded"] is True, "AI Models must be loaded!"

def inspect_file(filepath):
    with open(filepath, "rb") as f:
        b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")
    return client.post("/api/inspect", json={"image": b64}).get_json()

print("\n2. TEST CASE A: CLEAN METAL SURFACE")
clean_img = os.path.join(BASE_DIR, "dataset", "cracks", "no_crack", "clean_metal_0000.jpg")
res_clean = inspect_file(clean_img)
print("   Detected Material :", res_clean["material"]["predicted_class"])
print("   Material Conf     :", round(res_clean["material"]["confidence"] * 100, 2), "%")
print("   Crack Verdict     :", res_clean["crack"]["predicted_class"])
print("   Crack Conf        :", round(res_clean["crack"]["confidence"] * 100, 2), "%")
print("   Surface Condition :", res_clean["crack"]["surface_condition"])
print("   Inspection Status :", res_clean["record"]["status"])

print("\n3. TEST CASE B: DEFECTIVE CRACKED METAL SURFACE")
crack_img = os.path.join(BASE_DIR, "dataset", "cracks", "crack", "crack_0000.jpg")
res_crack = inspect_file(crack_img)
print("   Detected Material :", res_crack["material"]["predicted_class"])
print("   Material Conf     :", round(res_crack["material"]["confidence"] * 100, 2), "%")
print("   Crack Verdict     :", res_crack["crack"]["predicted_class"])
print("   Crack Conf        :", round(res_crack["crack"]["confidence"] * 100, 2), "%")
print("   Surface Condition :", res_crack["crack"]["surface_condition"])
print("   Inspection Status :", res_crack["record"]["status"])

print("\n4. TEST CASE C: PLASTIC COMPONENT (GATE CHECK)")
plastic_img = os.path.join(BASE_DIR, "dataset", "material", "plastic", "plastic_0000.jpg")
res_plastic = inspect_file(plastic_img)
print("   Detected Material :", res_plastic["material"]["predicted_class"])
print("   Material Conf     :", round(res_plastic["material"]["confidence"] * 100, 2), "%")
print("   Crack Verdict     :", res_plastic["crack"]["predicted_class"])
print("   Gating Logic Note :", res_plastic["crack"]["note"])
print("   Inspection Status :", res_plastic["record"]["status"])

print("\n5. TEST CASE D: OTHER / WOOD / UNKNOWN COMPONENT")
other_img = os.path.join(BASE_DIR, "dataset", "material", "other", "other_0000.jpg")
res_other = inspect_file(other_img)
print("   Detected Material :", res_other["material"]["predicted_class"])
print("   Material Conf     :", round(res_other["material"]["confidence"] * 100, 2), "%")
print("   Crack Verdict     :", res_other["crack"]["predicted_class"])
print("   Gating Logic Note :", res_other["crack"]["note"])
print("   Inspection Status :", res_other["record"]["status"])

print("\n6. SUMMARY DATABASE STATISTICS:")
stats = client.get("/api/statistics").get_json()
print("   Total Inspections :", stats["total_inspections"])
print("   Metal Count       :", stats["metal_inspections"])
print("   Cracks Detected   :", stats["cracks_detected"])
print("   Normal Surfaces   :", stats["normal_surfaces"])

print("\n7. SQLITE AUDIT TRAIL:")
history = client.get("/api/history").get_json()
print("   Saved Audit Rows  :", len(history))
for r in history[:3]:
    print(f"    - {r['inspection_code']} | {r['detected_material']} ({r['material_confidence']}%) | {r['crack_result']}")

print("\nALL VERIFICATION TESTS COMPLETED SUCCESSFULLY!")

