"""Slack entity schemas with rich context extraction for enhanced vectorization.

This implementation captures comprehensive context for Slack messages including:
- Thread context: Parent messages and reply summaries
- User enrichment: Real names and job titles
- Channel enrichment: Topics and purposes
- Rich content: Reactions, attachments, files with human-readable summaries
- Text cleaning: Resolves Slack markup to readable format
- Social signals: Reply counts, engagement metrics, pinned status
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import computed_field, field_validator

from airweave.platform.entities._airweave_field import AirweaveField
from airweave.platform.entities._base import BaseEntity


class SlackMessageEntity(BaseEntity):
    """Schema for Slack message entities from federated search.

    Reference:
        https://api.slack.com/methods/search.messages
    """

    # Base fields are inherited and set during entity creation:
    # - entity_id (message IID or timestamp)
    # - breadcrumbs (channel breadcrumb)
    # - name (from text preview)
    # - created_at (from timestamp)
    # - updated_at (None - messages don't have update timestamp)

    # API fields
    text: str = AirweaveField(
        ..., description="The text content of the message", embeddable=True, is_name=True
    )
    user: Optional[str] = AirweaveField(
        None, description="User ID of the message author", embeddable=False
    )
    username: Optional[str] = AirweaveField(
        None, description="Username of the message author", embeddable=True
    )
    ts: str = AirweaveField(
        ...,
        description="Message timestamp (unique identifier)",
        embeddable=False,
        is_entity_id=True,
    )
    channel_id: str = AirweaveField(
        ..., description="ID of the channel containing this message", embeddable=False
    )
    channel_name: Optional[str] = AirweaveField(
        None, description="Name of the channel", embeddable=True
    )
    channel_is_private: Optional[bool] = AirweaveField(
        None, description="Whether the channel is private", embeddable=False
    )
    type: str = AirweaveField(
        default="message", description="Type of the message", embeddable=False
    )
    permalink: Optional[str] = AirweaveField(
        None, description="Permalink to the message in Slack", embeddable=False
    )
    team: Optional[str] = AirweaveField(None, description="Team/workspace ID", embeddable=False)
    previous_message: Optional[Dict[str, Any]] = AirweaveField(
        None, description="Previous message for context", embeddable=False
    )
    next_message: Optional[Dict[str, Any]] = AirweaveField(
        None, description="Next message for context", embeddable=False
    )
    score: Optional[float] = AirweaveField(
        None, description="Search relevance score from Slack", embeddable=False
    )
    iid: Optional[str] = AirweaveField(None, description="Internal search ID", embeddable=False)
    url: Optional[str] = AirweaveField(
        None, description="URL to view the message in Slack", embeddable=False
    )
    message_time: datetime = AirweaveField(
        ..., description="Timestamp converted to datetime for hashing checks.", is_created_at=True
    )
    web_url_value: Optional[str] = AirweaveField(
        None, description="Permalink to open the message.", embeddable=False, unhashable=True
    )

    # Thread Context Fields
    thread_ts: Optional[str] = AirweaveField(
        None, description="Parent thread timestamp (Slack's thread identifier)", embeddable=False
    )
    thread_context: Optional[str] = AirweaveField(
        None,
        description="Concatenated text from parent + key replies for context",
        embeddable=True,
    )
    is_thread_parent: Optional[bool] = AirweaveField(
        None, description="Is this message the start of a thread?", embeddable=False
    )
    reply_count: Optional[int] = AirweaveField(
        None, description="How many replies? (importance signal)", embeddable=False
    )
    reply_users: Optional[List[str]] = AirweaveField(
        None, description="Usernames who replied (shows engagement)", embeddable=True
    )

    # Rich Content Fields
    attachments: Optional[List[Dict[str, Any]]] = AirweaveField(
        None, description="Raw attachment objects from API", embeddable=False
    )
    attachment_summary: Optional[str] = AirweaveField(
        None, description="Human-readable attachment summary", embeddable=True
    )
    reactions: Optional[Dict[str, int]] = AirweaveField(
        None, description="Reaction counts by emoji name", embeddable=False
    )
    reaction_summary: Optional[str] = AirweaveField(
        None, description="Human-readable reaction summary", embeddable=True
    )
    has_files: Optional[bool] = AirweaveField(
        None, description="Quick check if files present", embeddable=False
    )
    files_summary: Optional[str] = AirweaveField(
        None, description="Human-readable files summary with types", embeddable=True
    )

    # Message Metadata Fields
    is_pinned: Optional[bool] = AirweaveField(
        None, description="Pinned messages are important", embeddable=False
    )
    is_starred: Optional[bool] = AirweaveField(
        None, description="User marked as important", embeddable=False
    )
    is_bot_message: Optional[bool] = AirweaveField(
        None, description="Bot vs human distinction", embeddable=False
    )
    subtype: Optional[str] = AirweaveField(
        None, description="Message subtype (file_share, thread_broadcast, etc.)", embeddable=False
    )

    # User Enrichment Fields
    user_real_name: Optional[str] = AirweaveField(
        None, description="Full name from user profile", embeddable=True
    )
    user_title: Optional[str] = AirweaveField(None, description="Job title like", embeddable=True)

    # Channel Enrichment Fields
    channel_topic: Optional[str] = AirweaveField(
        None, description="Short channel topic", embeddable=True
    )
    channel_purpose: Optional[str] = AirweaveField(
        None, description="Longer channel description", embeddable=True
    )
    channel_member_count: Optional[int] = AirweaveField(
        None, description="Size indicates importance/scope", embeddable=False
    )

    # Mention Extraction Fields
    mentioned_users: Optional[List[str]] = AirweaveField(
        None, description="List of mentioned usernames", embeddable=True
    )
    mentioned_channels: Optional[List[str]] = AirweaveField(
        None, description="List of mentioned channel names", embeddable=True
    )

    # Cleaned Text Field
    cleaned_text: Optional[str] = AirweaveField(
        None,
        description="Message with resolved mentions/formatting (human-readable)",
        embeddable=True,
    )

    # Team/Workspace Context Fields
    team_name: Optional[str] = AirweaveField(
        None, description="Workspace name like 'Acme Corp'", embeddable=True
    )
    team_domain: Optional[str] = AirweaveField(
        None, description="Workspace domain like 'acme-corp.slack.com'", embeddable=False
    )

    @field_validator("cleaned_text")
    @classmethod
    def validate_cleaned_text(cls, v: Optional[str]) -> Optional[str]:
        """Ensure cleaned text doesn't have Slack markup."""
        if v and ("<@" in v or "<#" in v or "<http" in v):
            # Log warning but don't fail - this is informational
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Cleaned text still contains Slack markup: {v[:100]}")
        return v

    @field_validator("reply_count")
    @classmethod
    def validate_reply_count(cls, v: Optional[int]) -> Optional[int]:
        """Ensure reply count is non-negative."""
        if v is not None and v < 0:
            return 0
        return v

    @computed_field(return_type=str)
    def web_url(self) -> str:
        """Permalink for the Slack message."""
        return self.web_url_value or self.url or ""

    @computed_field(return_type=str)
    def rich_content(self) -> str:  # noqa: C901
        """Complete context - useful for debugging and full understanding.

        Combines all available context into a comprehensive text block showing
        workspace, channel, author, thread context, message, mentions, attachments,
        and engagement signals.
        """
        parts = []

        # Workspace context
        if self.team_name:
            parts.append(f"Workspace: {self.team_name}")

        # Channel context with topic
        channel_info = (
            f"Channel: #{self.channel_name}" if self.channel_name else f"Channel: {self.channel_id}"
        )
        if self.channel_topic:
            channel_info += f" ({self.channel_topic})"
        elif self.channel_purpose:
            # Truncate purpose if used as fallback
            purpose = (
                self.channel_purpose[:100] + "..."
                if len(self.channel_purpose) > 100
                else self.channel_purpose
            )
            channel_info += f" ({purpose})"
        parts.append(channel_info)

        # Author context
        author = self.user_real_name or self.username or "Unknown"
        if self.user_title:
            author += f" - {self.user_title}"
        parts.append(f"From: {author}")

        # Thread context
        if self.thread_context:
            parts.append(f"Thread Context: {self.thread_context}")

        # Main message
        message_text = self.cleaned_text or self.text
        if message_text:
            parts.append(f"Message: {message_text}")

        # Mentions
        if self.mentioned_users:
            parts.append(f"Mentions: {', '.join(self.mentioned_users)}")

        # Attachments
        if self.attachment_summary:
            parts.append(f"Attachments: {self.attachment_summary}")

        # Files
        if self.files_summary:
            parts.append(f"Files: {self.files_summary}")

        # Engagement summary
        engagement_parts = []
        if self.reaction_summary:
            engagement_parts.append(self.reaction_summary)
        if self.reply_count and self.reply_count > 0:
            engagement_parts.append(f"{self.reply_count} replies")
        if self.is_pinned:
            engagement_parts.append("pinned")
        if engagement_parts:
            parts.append(f"Engagement: {', '.join(engagement_parts)}")

        return "\n".join(parts)

    @computed_field(return_type=str)
    def embedding_text(self) -> str:
        """Token-optimized version - balances context with efficiency.

        This is what actually gets embedded. Concise but informative, targeting
        under 300 tokens total. Format is more compact than rich_content.
        """
        parts = []

        # Channel name (no topic to save tokens)
        if self.channel_name:
            parts.append(f"#{self.channel_name}")

        # Author name
        author = self.user_real_name or self.username
        if author:
            parts.append(f"@{author}")

        # Thread context (truncated)
        if self.thread_context:
            thread_text = self.thread_context
            if len(thread_text) > 200:
                thread_text = thread_text[:197] + "..."
            parts.append(f"[Thread] {thread_text}")

        # Main message (use cleaned text)
        message_text = self.cleaned_text or self.text
        if message_text:
            # Truncate very long messages
            if len(message_text) > 2000:
                message_text = message_text[:1997] + "..."
            parts.append(message_text)

        # Files (if present)
        if self.files_summary:
            # Truncate file summary
            file_text = self.files_summary
            if len(file_text) > 200:
                file_text = file_text[:197] + "..."
            parts.append(f"[Files: {file_text}]")

        # High engagement signal
        if self.reply_count and self.reply_count > 5:
            parts.append(f"[{self.reply_count} replies]")

        # Join with spaces for compact representation
        return " ".join(parts)
