
from rest_framework.response import Response
from blog.posts.models import Post
from blog.posts.serializer import PostSerializer


class PostRepository(object):
    def __init__(self):
        pass

    def get_all(self):
        serializer = PostSerializer(Post.objects.all(), many=True)
        return Response(serializer.data)

    def find_by_id(self):
        return