from rest_framework.response import Response

from shop.deliveries.models import Delivery
from shop.deliveries.serializer import DeliverySerializer


class DeliveryRepository(object):
    def __init__(self):
        pass

    def get_all(self):
        serializer = DeliverySerializer(Delivery.objects.all(), many=True)
        return Response(serializer.data)

    def find_by_id(self):
        return