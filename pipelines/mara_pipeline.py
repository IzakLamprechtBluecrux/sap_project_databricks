from pathlib import Path
import sys

# This file lives in: sap_project_databricks/pipelines/mara_pipeline.py
# parents[1] returns: sap_project_databricks/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(PROJECT_ROOT / "data_contract_pipeline"))

from data_contract_pipeline import run_data_contract_pipeline

CONTRACT_PATH = PROJECT_ROOT / "data_contracts" / "mara_contract.yml"

run_data_contract_pipeline(
    spark=spark,
    contract_path=str(CONTRACT_PATH),
    write_mode="overwrite"
)
