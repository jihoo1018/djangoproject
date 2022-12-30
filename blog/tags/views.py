from rest_framework.parsers import JSONParser
from rest_framework.decorators import api_view, parser_classes

from blog.tags.repositories import TagRepository
from blog.tags.serializer import TagSerializer

@api_view(['POST','PUT','PATCH','DELETE','GET'])
@parser_classes([JSONParser])
def tag_view(request):
    if request.method == "POST":
        return TagSerializer().create(request.data)
    elif request.method == "PATCH":
        return None
    elif request.method == "GET":
        return TagRepository().find_by_id()
    elif request.method == "PUT":
        return TagSerializer().update(request.data)
    elif request.method == "DELETE":
        return TagSerializer().delete(request.data)


@api_view(['GET'])
@parser_classes([JSONParser])
def tags_list(request):
    return TagRepository().get_all()
