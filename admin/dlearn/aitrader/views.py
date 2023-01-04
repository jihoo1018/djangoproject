from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser

from admin.dlearn.aitrader.services import TraderService


@api_view(['GET'])
@parser_classes([JSONParser])
def kospi(request):
    TraderService().manual()
    print(f'Enter Stroke with {request}')
    return JsonResponse({'Response Training': 'SUCCESS'})