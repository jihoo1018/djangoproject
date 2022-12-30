from rest_framework.response import Response

from multiplex.cinemas.models import Cinema
from multiplex.cinemas.serializer import CinemaSerializer


class CinemaRepository(object):
    def __init__(self):
        pass

    def get_all(self):
        serializer = CinemaSerializer(Cinema.objects.all(), many=True)
        return Response(serializer.data)

    def find_by_id(self):
        return