from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser


@api_view(['POST'])
@parser_classes([JSONParser])
def korean_classify(request):
    return JsonResponse({'result': ''})