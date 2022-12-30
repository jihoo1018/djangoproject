from rest_framework.parsers import JSONParser
from rest_framework.decorators import api_view, parser_classes

from shop.deliveries.repositories import DeliveryRepository
from shop.deliveries.serializer import DeliverySerializer


@api_view(['POST','PUT','PATCH','DELETE','GET'])
@parser_classes([JSONParser])
def delivery_view(request):
    if request.method == "POST":
        return DeliverySerializer().create(request.data)
    elif request.method == "PATCH":
        return None
    elif request.method == "GET":
        return DeliveryRepository().find_by_id(request.data)
    elif request.method == "PUT":
        return DeliverySerializer().update(request.data)
    elif request.method == "DELETE":
        return DeliverySerializer().delete(request.data)

@api_view(['GET'])
@parser_classes([JSONParser])
def delivery_list(request):
    return DeliveryRepository().get_all()


