from rest_framework.response import Response

from multiplex.movies.models import Movie
from multiplex.movies.serializer import MovieSerializer


class MoviesRepository(object):
    def __init__(self):
        pass

    def get_all(self):
        serializer = MovieSerializer(Movie.objects.all(), many=True)
        return Response(serializer.data)

    def find_by_id(self):
        return