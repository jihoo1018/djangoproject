from rest_framework.parsers import JSONParser
from rest_framework.decorators import api_view, parser_classes

from blog.posts.repositories import PostRepository
from blog.posts.serializer import PostSerializer

@api_view(['POST','PUT','PATCH','DELETE','GET'])
@parser_classes([JSONParser])
def post_view(request):
    if request.method == "POST":
        return PostSerializer().create(request.data)
    elif request.method == "PATCH":
        return None
    elif request.method == "GET":
        return PostRepository().find_by_id()
    elif request.method == "PUT":
        return PostSerializer().update(request.data)
    elif request.method == "DELETE":
        return PostSerializer().delete(request.data)


@api_view(['GET'])
@parser_classes([JSONParser])
def post_list(request):
    return PostRepository().get_all()


