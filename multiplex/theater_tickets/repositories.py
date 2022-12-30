from rest_framework.response import Response

from multiplex.theater_tickets.models import TheaterTicket
from multiplex.theater_tickets.serializer import TheaterTicketSerializer


class TheaterTicketsRepository(object):
    def __init__(self):
        pass

    def get_all(self):
        serializer = TheaterTicketSerializer(TheaterTicket.objects.all(), many=True)
        return Response(serializer.data)

    def find_by_id(self):
        return