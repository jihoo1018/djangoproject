from rest_framework.parsers import JSONParser
from rest_framework.decorators import api_view, parser_classes

from multiplex.theaters.repositories import TheatersRepository
from multiplex.theaters.serializer import TheaterSerializer


@api_view(['POST','PUT','PATCH','DELETE','GET'])
@parser_classes([JSONParser])
def theaters_view(request):
    if request.method == "POST":
        return TheaterSerializer().create(request.data)
    elif request.method == "PATCH":
        return None
    elif request.method == "GET":
        return TheatersRepository().find_by_id(request.data)
    elif request.method == "PUT":
        return TheaterSerializer().update(request.data)
    elif request.method == "DELETE":
        return TheaterSerializer().delete(request.data)


@api_view(['GET'])
@parser_classes([JSONParser])
def theater_list(request):
    return TheatersRepository().get_all()
