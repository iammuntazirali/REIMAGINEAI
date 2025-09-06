import praw

# Fill these with your actual Reddit app credentials
CLIENT_ID = "DhueH0nghtuJdSZwH61Z8Q"
CLIENT_SECRET = "mKJHxCHGFbvyUvCsb4jrjkIpBLhGeg"
USER_AGENT = "communityPlatformBot/1.0 by vermaapurva33"

reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

print("Read-only mode:", reddit.read_only)

# Example: print top 5 hot posts from r/python
for submission in reddit.subreddit("python").hot(limit=5):
    print(submission.title)
