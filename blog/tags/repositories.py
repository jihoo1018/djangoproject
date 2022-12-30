from rest_framework.response import Response

from blog.tags.models import Tag
from blog.tags.serializer import TagSerializer


class TagRepository(object):
    def __init__(self):
        pass

    def get_all(self):
        serializer = TagSerializer(Tag.objects.all(), many=True)
        return Response(serializer.data)

    def find_by_id(self):
        return