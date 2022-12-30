from rest_framework.parsers import JSONParser
from rest_framework.decorators import api_view, parser_classes

from blog.comments.repositories import CommentRepository
from blog.comments.serializer import CommentSerializer

@api_view(['POST','PUT','PATCH','DELETE','GET'])
@parser_classes([JSONParser])
def comment_view(request):
    if request.method == "POST":
        return CommentSerializer().create(request.data)
    elif request.method == "PATCH":
        return None
    elif request.method == "GET":
        return CommentRepository().find_by_id()
    elif request.method == "PUT":
        return CommentSerializer().update(request.data)
    elif request.method == "DELETE":
        return CommentSerializer().delete(request.data)



@api_view(['GET'])
@parser_classes([JSONParser])
def comment_list(request):
    return CommentRepository().get_all()
