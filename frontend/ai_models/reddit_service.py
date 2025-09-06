import os, logging
from typing import List, Dict
import praw

class RedditService:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT"),
            username=os.getenv("REDDIT_USERNAME"),
            password=os.getenv("REDDIT_PASSWORD")
        )
    def get_hot_posts(self, subreddit: str, limit: int) -> List[Dict]:
        posts=[]
        for s in self.reddit.subreddit(subreddit).hot(limit=limit):
            posts.append({
                "id": s.id, "title": s.title, "selftext": s.selftext,
                "score": s.score, "url": s.url,
                "created_utc": s.created_utc,
                "num_comments": s.num_comments,
                "author": str(s.author) if s.author else "deleted"
            })
        return posts
    def search_posts(self, query: str, subreddit: str, limit: int) -> List[Dict]:
        posts=[]
        for s in self.reddit.subreddit(subreddit).search(query, limit=limit):
            posts.append({
                "id": s.id, "title": s.title, "selftext": s.selftext,
                "score": s.score, "url": s.url,
                "created_utc": s.created_utc,
                "relevance_score": s.score
            })
        return posts
    def get_detailed_hot_posts(self, subreddit: str, limit: int):
        posts_data = []
        subreddit_obj = self.reddit.subreddit(subreddit)
        for post in subreddit_obj.hot(limit=limit):
            post.comments.replace_more(limit=0)  # load all comments efficiently
            comments = [comment.body for comment in post.comments.list()[:10]]  # first 10 comments
            posts_data.append({
                "id": post.id,
                "title": post.title,
                "selftext": post.selftext,
                "score": post.score,
                "url": post.url,
                "author": str(post.author),
                "num_comments": post.num_comments,
                "created_utc": post.created_utc,
                "comments": comments,
            })
        return posts_data
    def _extract_comments(self, comments, depth=0):
        comment_list = []
        for comment in comments:
            comment_list.append({
                "id": comment.id,
                "author": str(comment.author),
                "body": comment.body,
                "score": comment.score,
                "created_utc": comment.created_utc,
                "depth": depth,
                "replies": self._extract_comments(comment.replies, depth + 1) if comment.replies else [],
            })
        return comment_list

    def get_post_with_comments(self, post_id: str):
        submission = self.reddit.submission(id=post_id)
        submission.comments.replace_more(limit=2)
        comments = self._extract_comments(submission.comments)
        return {
            "id": submission.id,
            "title": submission.title,
            "selftext": submission.selftext,
            "author": str(submission.author),
            "score": submission.score,
            "num_comments": submission.num_comments,
            "created_utc": submission.created_utc,
            "media": submission.media,
            "gallery_data": getattr(submission, "gallery_data", None),
            "comments": comments,
            "url": submission.url,
            "thumbnail": submission.thumbnail if submission.thumbnail != "self" else None,
            "is_video": getattr(submission, "is_video", False),
        }



import os
import praw

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    username=os.getenv("REDDIT_USERNAME"),
    password=os.getenv("REDDIT_PASSWORD"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)

try:
    subreddit = reddit.subreddit("all")
    print("Fetching 5 hot posts from r/all...")
    for post in subreddit.hot(limit=5):
        print(f"Title: {post.title}")
        print(f"ID: {post.id}")
        print(f"Score: {post.score}")
        print(f"URL: {post.url}")
        print("-" * 40)
except Exception as e:
    print("Error accessing Reddit API:", e)

# check_reddit_api.py

# import os
# import praw

# def main():
#     # Print environment variables to verify they are loaded
#     print("Reddit API Credentials from environment variables:")
#     print("REDDIT_CLIENT_ID:", os.getenv("REDDIT_CLIENT_ID"))
#     print("REDDIT_CLIENT_SECRET:", os.getenv("REDDIT_CLIENT_SECRET"))
#     print("REDDIT_USERNAME:", os.getenv("REDDIT_USERNAME"))
#     print("REDDIT_USER_AGENT:", os.getenv("REDDIT_USER_AGENT"))
#     print("-" * 40)

#     # Initialize Reddit instance with credentials from environment variables
#     try:
#         reddit = praw.Reddit(
#             client_id=os.getenv("REDDIT_CLIENT_ID"),
#             client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
#             username=os.getenv("REDDIT_USERNAME"),
#             password=os.getenv("REDDIT_PASSWORD"),
#             user_agent=os.getenv("REDDIT_USER_AGENT"),
#         )
#     except Exception as e:
#         print("Error initializing PRAW Reddit instance:", e)
#         return

#     # Test API by fetching top 5 hot posts from r/all
#     try:
#         print("Fetching 5 hot posts from r/all:")
#         subreddit = reddit.subreddit("all")
#         for post in subreddit.hot(limit=5):
#             print(f"Title: {post.title}")
#             print(f"ID: {post.id}")
#             print(f"Score: {post.score}")
#             print(f"URL: {post.url}")
#             print("-" * 40)
#     except Exception as e:
#         print("Error fetching posts from Reddit API:", e)

# if __name__ == "__main__":
#     main()
