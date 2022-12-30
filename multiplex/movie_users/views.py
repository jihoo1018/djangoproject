from rest_framework.parsers import JSONParser
from rest_framework.decorators import api_view, parser_classes

from multiplex.movie_users.repositories import MovieUserRepository
from multiplex.movie_users.serializer import MovieUserSerializer


@api_view(['POST','PUT','PATCH','DELETE','GET'])
@parser_classes([JSONParser])
def movie_user_view(request):
    if request.method == "POST":
        return MovieUserSerializer().create(request.data)
    elif request.method == "PATCH":
        return None
    elif request.method == "GET":
        return MovieUserRepository().find_by_id(request.data)
    elif request.method == "PUT":
        return MovieUserSerializer().update(request.data)
    elif request.method == "DELETE":
        return MovieUserSerializer().delete(request.data)




@api_view(['GET'])
@parser_classes([JSONParser])
def user_list(request):
    return MovieUserRepository().get_all()


