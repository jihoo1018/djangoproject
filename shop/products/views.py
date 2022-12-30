from rest_framework.parsers import JSONParser
from rest_framework.decorators import api_view, parser_classes

from shop.products.repositories import ProductRepository
from shop.products.serializer import ProductSerializer


@api_view(['POST','PUT','PATCH','DELETE','GET'])
@parser_classes([JSONParser])
def product_view(request):
    if request.method == "POST":
        return ProductSerializer().create(request.data)
    elif request.method == "PATCH":
        return None
    elif request.method == "GET":
        return ProductRepository().find_by_id(request.data)
    elif request.method == "PUT":
        return ProductSerializer().update(request.data)
    elif request.method == "DELETE":
        return ProductSerializer().delete(request.data)

@api_view(['GET'])
@parser_classes([JSONParser])
def product_list(request):
    return ProductRepository().get_all()

