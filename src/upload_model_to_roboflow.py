from roboflow import Roboflow

rf = Roboflow(api_key="ZKM1SNXnj5Dh014x3TVx")
workspace = rf.workspace("mainarassegdataset3")

workspace.deploy_model(
  model_type="yolov11",
  model_path=r"C:\Users\LENOVO\Desktop\Segmentation paper DG",
  project_ids=["cadis_final"],
  model_name="my-custom-model"
)

