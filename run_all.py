from src.get_holdings_universe import main as build_universe
from src.download_data import main as download_prices
from src.build_panel import main as build_panel
from src.train_model import main as train_model
from src.update_model_readme import main as update_readme

if __name__ == "__main__":
    build_universe()
    download_prices()
    build_panel()
    train_model()
    update_readme()
