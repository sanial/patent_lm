param(
    [string]$ConfigPath = "configs/pipeline.yaml"
)

$env:PYTHONPATH = "src"
python -m patent_pipeline.cli bootstrap
python -m patent_pipeline.cli all --config $ConfigPath
