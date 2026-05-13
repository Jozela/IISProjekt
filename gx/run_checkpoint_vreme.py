import sys
import great_expectations as gx

context = gx.get_context()

datasource_name = "vreme"
data_asset_name = "vreme_data"

asset = context.get_datasource(datasource_name).get_asset(data_asset_name)

checkpoint_name = "vreme_checkpoint"
checkpoint = context.get_checkpoint(checkpoint_name)

run_id = "vreme_run"
checkpoint_result = checkpoint.run(run_id=run_id)

context.build_data_docs()

if checkpoint_result["success"]:
    print("Validacija uspešna!")
    sys.exit(0)
else:
    print("Validacija neuspešna!")
    sys.exit(1)