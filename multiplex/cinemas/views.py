from rest_framework.parsers import JSONParser
from rest_framework.decorators import api_view, parser_classes

from multiplex.cinemas.repositories import CinemaRepository
from multiplex.cinemas.serializer import CinemaSerializer

@api_view(['POST','PUT','PATCH','DELETE','GET'])
@parser_classes([JSONParser])
def cinema_view(request):
    if request.method == "POST":
        return CinemaSerializer().create(request.data)
    elif request.method == "PATCH":
        return None
    elif request.method == "GET":
        return CinemaRepository().find_by_id(request.data)
    elif request.method == "PUT":
        return CinemaSerializer().update(request.data)
    elif request.method == "DELETE":
        return CinemaSerializer().delete(request.data)



@api_view(['GET'])
@parser_classes([JSONParser])
def cinema_list(request):
    return CinemaRepository().get_all()
