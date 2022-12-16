import json

from django.http import JsonResponse, QueryDict
from matplotlib import pyplot as plt
from rest_framework import serializers
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser
import tensorflow as tf

from admin.dlearn.fashion_service import FashionService


@api_view(["GET"])
def fashion(request):
    print(f"######## React getID is {request.GET['num']} ########")
    return JsonResponse({'result': FashionService().service_model(int(request.GET['num']))})

@api_view(["POST"])
def fashion(request):
    data = json.loads(request.body)  # json to dict
    print(f"######## React postID is {data['id']} ########")
    return JsonResponse({'result': FashionService().service_model(int(data['id']))})