from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .apps import StockpredictorConfig
import numpy as np
class call_model(APIView):

    def get(self,request):
        response = np.array(StockpredictorConfig.predictor.predict(),dtype=str)
        d = dict(enumerate(response.flatten(), 1))

        # returning JSON response
        return JsonResponse(d,safe=False)
        # if request.method == 'GET':
            
        #     # sentence is the query we want to get the prediction for
        #     # params =  request.GET.get()
            
        #     # predict method used to get the prediction