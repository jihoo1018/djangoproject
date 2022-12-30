from rest_framework.response import Response

from multiplex.theaters.models import Theater
from multiplex.theaters.serializer import TheaterSerializer


class TheatersRepository(object):
    def __init__(self):
        pass

    def get_all(self):
        serializer = TheaterSerializer(Theater.objects.all(), many=True)
        return Response(serializer.data)

    def find_by_id(self):
        return