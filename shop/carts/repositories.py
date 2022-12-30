from rest_framework.response import Response

from shop.carts.models import Cart
from shop.carts.serializer import CartSerializer


class CartRepository(object):
    def __init__(self):
        pass

    def get_all(self):
        serializer = CartSerializer(Cart.objects.all(), many=True)
        return Response(serializer.data)

    def find_by_id(self):
        return