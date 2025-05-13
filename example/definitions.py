from dagster import Definitions
from . import assets

defs = Definitions(
    assets=[
        assets.load_data,
        assets.eda,
        assets.prepare_data,
        assets.train_model,
        assets.evaluate_model,
        assets.analyze_residuals,
    ]
)