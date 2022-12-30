from rest_framework.response import Response

from multiplex.movie_users.models import MovieUser
from multiplex.movie_users.serializer import MovieUserSerializer


class MovieUserRepository(object):
    def __init__(self):
        pass

    def get_all(self):
        serializer = MovieUserSerializer(MovieUser.objects.all(), many=True)
        return Response(serializer.data)

    def find_by_id(self):
        return