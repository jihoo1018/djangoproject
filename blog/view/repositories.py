from rest_framework.response import Response

from blog.view.models import View
from blog.view.serializer import BlogViewSerializer


class BlogViewRepository(object):
    def __init__(self):
        pass

    def get_all(self):
        serializer = BlogViewSerializer(View.objects.all(), many=True)
        return Response(serializer.data)

    def find_by_id(self):
        return