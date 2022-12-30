from requests import Response

from blog.comments.models import Comment
from blog.comments.serializer import CommentSerializer


class CommentRepository(object):
    def __init__(self):
        pass

    def get_all(self):
        serializer = CommentSerializer(Comment.objects.all(), many=True)
        return Response(serializer.data)

    def find_by_id(self):
        return
