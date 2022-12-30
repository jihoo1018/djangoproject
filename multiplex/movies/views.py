from django.http import JsonResponse
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser

from multiplex.movies.repositories import MoviesRepository
from multiplex.movies.serializer import MovieSerializer
from multiplex.movies.services import DcGan


@api_view(['GET'])
@parser_classes([JSONParser])
def fake_face(request):
    DcGan().fake_images()
    print(f'Enter Show Faces with {request}')
    return JsonResponse({'Response Training': 'SUCCESS'})

@api_view(['POST','PUT','PATCH','DELETE','GET'])
@parser_classes([JSONParser])
def movies_view(request):
    if request.method == "POST":
        return MovieSerializer().create(request.data)
    elif request.method == "PATCH":
        return None
    elif request.method == "GET":
        return MoviesRepository().find_by_id(request.data)
    elif request.method == "PUT":
        return MovieSerializer().update(request.data)
    elif request.method == "DELETE":
        return MovieSerializer().delete(request.data)


@api_view(['GET'])
@parser_classes([JSONParser])
def movies_list(request):
    return MoviesRepository().get_all()



