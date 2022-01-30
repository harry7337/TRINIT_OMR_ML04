from django.apps import AppConfig
import html
import pathlib
from os import path as Path

from stockpredictor.StockPredictor import StockPredictor


class StockpredictorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'stockpredictor'
    predictor = StockPredictor()
