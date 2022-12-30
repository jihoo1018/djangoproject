from rest_framework.parsers import JSONParser
from rest_framework.decorators import api_view, parser_classes

from multiplex.showtimes.repositories import ShowtimesRepository
from multiplex.showtimes.serializer import ShowtimeSerializer

@api_view(['POST','PUT','PATCH','DELETE','GET'])
@parser_classes([JSONParser])
def showtimes_view(request):
    if request.method == "POST":
        return ShowtimeSerializer().create(request.data)
    elif request.method == "PATCH":
        return None
    elif request.method == "GET":
        return ShowtimesRepository().find_by_id(request.data)
    elif request.method == "PUT":
        return ShowtimeSerializer().update(request.data)
    elif request.method == "DELETE":
        return ShowtimeSerializer().delete(request.data)


@api_view(['GET'])
@parser_classes([JSONParser])
def showtimes_list(request):
    return ShowtimesRepository().get_all()


