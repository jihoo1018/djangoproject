from rest_framework.parsers import JSONParser
from rest_framework.decorators import api_view, parser_classes

from shop.categories.repositories import CategoryRepository
from shop.categories.serializer import CategorySerializer


@api_view(['POST','PUT','PATCH','DELETE','GET'])
@parser_classes([JSONParser])
def catgories_view(request):
    if request.method == "POST":
        return CategorySerializer().create(request.data)
    elif request.method == "PATCH":
        return None
    elif request.method == "GET":
        return CategoryRepository().find_by_id(request.data)
    elif request.method == "PUT":
        return CategorySerializer().update(request.data)
    elif request.method == "DELETE":
        return CategorySerializer().delete(request.data)

@api_view(['GET'])
@parser_classes([JSONParser])
def category_list(request):
    return CategoryRepository().get_all()

