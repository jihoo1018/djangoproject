from rest_framework.response import Response

from shop.categories.models import Category
from shop.categories.serializer import CategorySerializer


class CategoryRepository(object):
    def __init__(self):
        pass

    def get_all(self):
        serializer = CategorySerializer(Category.objects.all(), many=True)
        return Response(serializer.data)

    def find_by_id(self):
        return