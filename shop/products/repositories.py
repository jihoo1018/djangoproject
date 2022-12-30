from rest_framework.response import Response

from shop.products.models import Product
from shop.products.serializer import ProductSerializer


class ProductRepository(object):
    def __init__(self):
        pass

    def get_all(self):
        serializer = ProductSerializer(Product.objects.all(), many=True)
        return Response(serializer.data)

    def find_by_id(self):
        return