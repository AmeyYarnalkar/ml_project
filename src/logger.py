from pathlib import Path
from datetime import datetime
import logging

log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

log_dir = Path.cwd() / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

log_path = log_dir / log_file

logging.basicConfig(
    filename=str(log_path),
    format='[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

