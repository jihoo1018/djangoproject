from rest_framework.response import Response

from multiplex.showtimes.models import Showtime
from multiplex.showtimes.serializer import ShowtimeSerializer


class ShowtimesRepository(object):
    def __init__(self):
        pass

    def get_all(self):
        serializer = ShowtimeSerializer(Showtime.objects.all(), many=True)
        return Response(serializer.data)

    def find_by_id(self):
        return