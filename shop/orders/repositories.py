from rest_framework.response import Response

from shop.orders.models import Order
from shop.orders.serializer import OrderSerializer


class OrderRepository(object):
    def __init__(self):
        pass

    def get_all(self):
        serializer = OrderSerializer(Order.objects.all(), many=True)
        return Response(serializer.data)

    def find_by_id(self):
        return