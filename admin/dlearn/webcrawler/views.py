import json

from django.http import JsonResponse, QueryDict
from matplotlib import pyplot as plt
from rest_framework import serializers
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser
from admin.dlearn.webcrawler.services import ScrapService


@api_view(['GET'])
def navermovie(request):
    if request.method == 'GET':
        return JsonResponse(
            {'result': ScrapService().naver_movie_review()})
    else:
        print(f"######## ID is None ########")