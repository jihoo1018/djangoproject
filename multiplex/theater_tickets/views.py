from rest_framework.parsers import JSONParser
from rest_framework.decorators import api_view, parser_classes

from multiplex.theater_tickets.repositories import TheaterTicketsRepository
from multiplex.theater_tickets.serializer import TheaterTicketSerializer


@api_view(['POST','PUT','PATCH','DELETE','GET'])
@parser_classes([JSONParser])
def theater_ticket_view(request):
    if request.method == "POST":
        return TheaterTicketSerializer().create(request.data)
    elif request.method == "PATCH":
        return None
    elif request.method == "GET":
        return TheaterTicketsRepository().find_by_id(request.data)
    elif request.method == "PUT":
        return TheaterTicketSerializer().update(request.data)
    elif request.method == "DELETE":
        return TheaterTicketSerializer().delete(request.data)


@api_view(['GET'])
@parser_classes([JSONParser])
def theater_tickets_list(request):
    return TheaterTicketsRepository().get_all()


