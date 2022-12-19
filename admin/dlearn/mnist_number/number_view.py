import json

from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser

from admin.dlearn.mnist_number.number_service import NumberService


@api_view(['POST'])
@parser_classes([JSONParser])
def number(request):
        id = json.loads(request.body)  # json to dict
        print(f"######## POST id is {int(id)} ########")
        # a = NumberService().service_model(int(id))
        # print(f" 리턴결과 : {a} ")
        return JsonResponse({'result': id})