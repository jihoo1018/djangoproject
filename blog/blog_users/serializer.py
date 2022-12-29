from rest_framework import serializers
from blog.blog_users.models import BlogUser


class BlogUserSerializer(serializers.ModelSerializer):
    class Meta:
        model = BlogUser
        fields = '__all__'
