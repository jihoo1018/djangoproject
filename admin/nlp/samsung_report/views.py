from django.http import JsonResponse
from rest_framework.decorators import api_view

from admin.nlp.samsung_report.services import Service


@api_view(['GET'])
def konlp(request):
    if request.method == 'GET':
        return JsonResponse(
            {'result': Service().frequent_text()})
    else:
        print(f"######## ID is None ########")