from rest_framework import serializers
from blog.posts.models import Post


class PostSerializer(serializers.ModelSerializer):
    class Meta:
        model = Post
        fields = '__all__'

    def create(self, validated_data):
        return Post.objects.create(**validated_data)

    def update(self, instace, valicated_data):
        Post.objects.filter(pk=instace.id).update(**valicated_data)

    def delete(self, instance, valicated_data):
        pass
