from rest_framework.parsers import JSONParser
from rest_framework.decorators import api_view, parser_classes

from blog.view.repositories import BlogViewRepository
from blog.view.serializer import BlogViewSerializer

@api_view(['POST','PUT','PATCH','DELETE','GET'])
@parser_classes([JSONParser])
def views_view(request):
    if request.method == "POST":
        return BlogViewSerializer().create(request.data)
    elif request.method == "PATCH":
        return None
    elif request.method == "GET":
        return BlogViewRepository().find_by_id()
    elif request.method == "PUT":
        return BlogViewSerializer().update(request.data)
    elif request.method == "DELETE":
        return BlogViewSerializer().delete(request.data)


@api_view(['GET'])
@parser_classes([JSONParser])
def blog_view_list(request):
    return BlogViewRepository().get_all()

