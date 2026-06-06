from __future__ import annotations

from pathlib import Path
from typing import Any

from whispertome.agent.tools import (
    AgentTool,
    AgentToolRegistry,
    boolean_schema,
    integer_schema,
    nullable_string_schema,
    object_schema,
    string_array_schema,
    string_schema,
)
from whispertome.organizer.store import OrganizerStore
from whispertome.system.tools import build_system_control_tools


def build_organizer_store(project_root: Path) -> OrganizerStore:
    return OrganizerStore(project_root / "artifacts" / "organizer")


def build_organization_tool_registry(
    project_root: Path,
    *,
    store: OrganizerStore | None = None,
) -> AgentToolRegistry:
    active_store = store or build_organizer_store(project_root)
    return AgentToolRegistry(
        [
            _notes_add(active_store),
            _notes_list(active_store),
            _preferences_save(active_store),
            _preferences_list(active_store),
            _reminders_add(active_store),
            _reminders_list(active_store),
            _reminders_complete(active_store),
            _checklist_create(active_store),
            _checklist_append(active_store),
            _checklist_list(active_store),
            _checklist_complete(active_store),
            _itinerary_add(active_store),
            _itinerary_list(active_store),
            _tasks_add(active_store),
            _tasks_list(active_store),
            _tasks_complete(active_store),
            _projects_create(active_store),
            _projects_list(active_store),
            _time_sensitive_check(active_store),
            _daily_plan_get(active_store),
            _daily_plan_save(active_store),
            _decisions_add(active_store),
            _decisions_list(active_store),
            _people_add(active_store),
            _people_note_add(active_store),
            _people_list(active_store),
            _organization_summary(active_store),
            *build_system_control_tools(),
        ]
    )


def _notes_add(store: OrganizerStore) -> AgentTool:
    return AgentTool(
        name="notes_add",
        description="Save a loose note, idea, fact, dictated text, or memory.",
        parameters=object_schema(
            {
                "content": string_schema("The note body to save."),
                "title": nullable_string_schema("Short optional note title."),
                "tags": string_array_schema("Optional lowercase tags."),
            },
            required=["content"],
        ),
        handler=lambda args: {
            "ok": True,
            "note": store.add_note(
                content=str(args["content"]),
                title=_optional_text(args.get("title")),
                tags=_string_list(args.get("tags")),
            ),
        },
    )


def _notes_list(store: OrganizerStore) -> AgentTool:
    return AgentTool(
        name="notes_list",
        description="List saved notes, optionally filtered by query or tags.",
        parameters=object_schema(
            {
                "query": nullable_string_schema("Optional search text."),
                "tags": string_array_schema("Optional tags that must be present."),
                "limit": integer_schema("Maximum notes to return.", minimum=1),
            },
        ),
        handler=lambda args: {
            "ok": True,
            "notes": store.list_notes(
                query=_optional_text(args.get("query")),
                tags=_string_list(args.get("tags")),
                limit=_positive_int(args.get("limit"), default=10),
            ),
        },
    )


def _preferences_save(store: OrganizerStore) -> AgentTool:
    return AgentTool(
        name="preferences_save",
        description=(
            "Save or update a stable user preference/default for future turns, such as "
            "what to call the user, preferred units, tone, formatting, language, or "
            "other persistent assistant behavior. Do not use for one-off commands."
        ),
        parameters=object_schema(
            {
                "key": string_schema(
                    "Stable snake_case key, such as preferred_name, units, or temperature_unit."
                ),
                "category": nullable_string_schema(
                    "Short bucket such as identity, units, style, formatting, or defaults."
                ),
                "preference": string_schema(
                    "One compact sentence to inject later, e.g. 'Call the user Marcus.'"
                ),
                "value": nullable_string_schema(
                    "Optional normalized value, e.g. Marcus, metric, celsius, concise."
                ),
                "evidence": nullable_string_schema(
                    "Optional raw user wording that caused this preference."
                ),
                "priority": integer_schema(
                    (
                        "Prompt priority from 0 to 100. Use 80+ for identity "
                        "or safety-critical defaults."
                    ),
                    minimum=0,
                ),
            },
            required=["key", "preference"],
        ),
        handler=lambda args: {
            "ok": True,
            "preference": store.save_preference(
                key=str(args["key"]),
                category=_optional_text(args.get("category")),
                preference=str(args["preference"]),
                value=_optional_text(args.get("value")),
                evidence=_optional_text(args.get("evidence")),
                priority=_optional_int(args.get("priority")),
            ),
        },
    )


def _preferences_list(store: OrganizerStore) -> AgentTool:
    return AgentTool(
        name="preferences_list",
        description="List or search saved user preferences that are injected into future prompts.",
        parameters=object_schema(
            {
                "active_only": boolean_schema(
                    "Whether only active preferences should be returned."
                ),
                "query": nullable_string_schema("Optional search text."),
                "limit": integer_schema("Maximum preferences to return.", minimum=1),
            },
        ),
        handler=lambda args: {
            "ok": True,
            "preferences": store.list_preferences(
                active_only=bool(args.get("active_only", True)),
                query=_optional_text(args.get("query")),
                limit=_positive_int(args.get("limit"), default=20),
            ),
        },
    )


def _reminders_add(store: OrganizerStore) -> AgentTool:
    return AgentTool(
        name="reminders_add",
        description="Create a reminder or follow-up with an optional due date/time.",
        parameters=object_schema(
            {
                "title": string_schema("Reminder title."),
                "due_date": nullable_string_schema("ISO due date if known."),
                "due_time": nullable_string_schema("Local due time if known."),
                "notes": nullable_string_schema("Optional detail."),
                "priority": nullable_string_schema("Priority such as low, normal, high."),
                "tags": string_array_schema("Optional lowercase tags."),
            },
            required=["title"],
        ),
        handler=lambda args: {
            "ok": True,
            "reminder": store.add_reminder(
                title=str(args["title"]),
                due_date=_optional_text(args.get("due_date")),
                due_time=_optional_text(args.get("due_time")),
                notes=_optional_text(args.get("notes")),
                priority=_optional_text(args.get("priority")),
                tags=_string_list(args.get("tags")),
            ),
        },
    )


def _reminders_list(store: OrganizerStore) -> AgentTool:
    return AgentTool(
        name="reminders_list",
        description="List reminders, due reminders, or completed reminders.",
        parameters=object_schema(
            {
                "status": nullable_string_schema(
                    "Optional status filter, usually pending or done."
                ),
                "due_before": nullable_string_schema("ISO date upper bound for due reminders."),
                "include_completed": boolean_schema(
                    "Whether completed reminders should be returned."
                ),
                "query": nullable_string_schema("Optional search text."),
                "limit": integer_schema("Maximum reminders to return.", minimum=1),
            },
        ),
        handler=lambda args: {
            "ok": True,
            "reminders": store.list_reminders(
                status=_optional_text(args.get("status")),
                due_before=_optional_text(args.get("due_before")),
                include_completed=bool(args.get("include_completed", False)),
                query=_optional_text(args.get("query")),
                limit=_positive_int(args.get("limit"), default=10),
            ),
        },
    )


def _reminders_complete(store: OrganizerStore) -> AgentTool:
    return AgentTool(
        name="reminders_complete",
        description="Mark a reminder complete by id or matching title text.",
        parameters=object_schema(
            {
                "reminder_id": nullable_string_schema("Exact reminder id if known."),
                "title_query": nullable_string_schema("Reminder title search text."),
            },
        ),
        handler=lambda args: {
            "ok": True,
            "reminder": store.complete_reminder(
                reminder_id=_optional_text(args.get("reminder_id")),
                title_query=_optional_text(args.get("title_query")),
            ),
        },
    )


def _checklist_create(store: OrganizerStore) -> AgentTool:
    return AgentTool(
        name="checklist_create",
        description="Create a checklist with actionable items.",
        parameters=object_schema(
            {
                "title": string_schema("Checklist title."),
                "items": string_array_schema("Checklist item text."),
                "tags": string_array_schema("Optional lowercase tags."),
            },
            required=["title", "items"],
        ),
        handler=lambda args: {
            "ok": True,
            "checklist": store.create_checklist(
                title=str(args["title"]),
                items=_string_list(args.get("items")),
                tags=_string_list(args.get("tags")),
            ),
        },
    )


def _checklist_append(store: OrganizerStore) -> AgentTool:
    return AgentTool(
        name="checklist_append",
        description="Append new items to an existing checklist.",
        parameters=object_schema(
            {
                "checklist_id": nullable_string_schema("Exact checklist id if known."),
                "title_query": nullable_string_schema("Checklist title search text."),
                "items": string_array_schema("New checklist items to append."),
            },
            required=["items"],
        ),
        handler=lambda args: {
            "ok": True,
            **store.append_checklist_items(
                checklist_id=_optional_text(args.get("checklist_id")),
                title_query=_optional_text(args.get("title_query")),
                items=_string_list(args.get("items")),
            ),
        },
    )


def _checklist_list(store: OrganizerStore) -> AgentTool:
    return AgentTool(
        name="checklist_list",
        description="List checklists and their items.",
        parameters=object_schema(
            {
                "include_completed": boolean_schema("Whether completed items should be included."),
                "query": nullable_string_schema("Optional checklist title search text."),
                "limit": integer_schema("Maximum checklists to return.", minimum=1),
            },
        ),
        handler=lambda args: {
            "ok": True,
            "checklists": store.list_checklists(
                include_completed=bool(args.get("include_completed", True)),
                query=_optional_text(args.get("query")),
                limit=_positive_int(args.get("limit"), default=10),
            ),
        },
    )


def _checklist_complete(store: OrganizerStore) -> AgentTool:
    return AgentTool(
        name="checklist_complete",
        description="Mark a checklist item complete by id or matching text.",
        parameters=object_schema(
            {
                "checklist_id": nullable_string_schema("Exact checklist id if known."),
                "title_query": nullable_string_schema("Checklist title search text."),
                "item_id": nullable_string_schema("Exact checklist item id if known."),
                "item_text": nullable_string_schema("Checklist item search text."),
            },
        ),
        handler=lambda args: {
            "ok": True,
            **store.complete_checklist_item(
                checklist_id=_optional_text(args.get("checklist_id")),
                title_query=_optional_text(args.get("title_query")),
                item_id=_optional_text(args.get("item_id")),
                item_text=_optional_text(args.get("item_text")),
            ),
        },
    )


def _itinerary_add(store: OrganizerStore) -> AgentTool:
    return AgentTool(
        name="itinerary_add",
        description="Add a dated or undated itinerary entry, appointment, stop, or trip item.",
        parameters=object_schema(
            {
                "title": string_schema("Event, stop, appointment, or itinerary item title."),
                "date": nullable_string_schema("ISO date if known, such as 2026-06-06."),
                "start_time": nullable_string_schema("Local start time if known, such as 14:30."),
                "end_time": nullable_string_schema("Local end time if known."),
                "location": nullable_string_schema("Place, address, or meeting location."),
                "notes": nullable_string_schema("Optional extra details."),
                "category": nullable_string_schema("Optional type such as travel, meal, meeting."),
            },
            required=["title"],
        ),
        handler=lambda args: {
            "ok": True,
            "itinerary_item": store.add_itinerary_item(
                title=str(args["title"]),
                date=_optional_text(args.get("date")),
                start_time=_optional_text(args.get("start_time")),
                end_time=_optional_text(args.get("end_time")),
                location=_optional_text(args.get("location")),
                notes=_optional_text(args.get("notes")),
                category=_optional_text(args.get("category")),
            ),
        },
    )


def _itinerary_list(store: OrganizerStore) -> AgentTool:
    return AgentTool(
        name="itinerary_list",
        description="List itinerary entries, optionally filtered by date range or search text.",
        parameters=object_schema(
            {
                "date_from": nullable_string_schema("Start ISO date, inclusive."),
                "date_to": nullable_string_schema("End ISO date, inclusive."),
                "query": nullable_string_schema("Optional search text."),
                "limit": integer_schema("Maximum itinerary items to return.", minimum=1),
            },
        ),
        handler=lambda args: {
            "ok": True,
            "itinerary": store.list_itinerary(
                date_from=_optional_text(args.get("date_from")),
                date_to=_optional_text(args.get("date_to")),
                query=_optional_text(args.get("query")),
                limit=_positive_int(args.get("limit"), default=10),
            ),
        },
    )


def _tasks_add(store: OrganizerStore) -> AgentTool:
    return AgentTool(
        name="tasks_add",
        description="Create a single task with optional due date, priority, project, and tags.",
        parameters=object_schema(
            {
                "title": string_schema("Task title."),
                "notes": nullable_string_schema("Optional task notes."),
                "due_date": nullable_string_schema("ISO due date if known."),
                "due_time": nullable_string_schema("Local due time if known."),
                "priority": nullable_string_schema("Priority such as low, normal, high."),
                "project_id": nullable_string_schema("Project id if this task belongs to one."),
                "tags": string_array_schema("Optional lowercase tags."),
            },
            required=["title"],
        ),
        handler=lambda args: {
            "ok": True,
            "task": store.add_task(
                title=str(args["title"]),
                notes=_optional_text(args.get("notes")),
                due_date=_optional_text(args.get("due_date")),
                due_time=_optional_text(args.get("due_time")),
                priority=_optional_text(args.get("priority")),
                project_id=_optional_text(args.get("project_id")),
                tags=_string_list(args.get("tags")),
            ),
        },
    )


def _tasks_list(store: OrganizerStore) -> AgentTool:
    return AgentTool(
        name="tasks_list",
        description="List tasks by status, project, due date, or search text.",
        parameters=object_schema(
            {
                "status": nullable_string_schema("Optional status filter, usually open or done."),
                "project_id": nullable_string_schema("Project id filter."),
                "due_before": nullable_string_schema("ISO date upper bound for due tasks."),
                "include_completed": boolean_schema("Whether completed tasks should be returned."),
                "query": nullable_string_schema("Optional search text."),
                "limit": integer_schema("Maximum tasks to return.", minimum=1),
            },
        ),
        handler=lambda args: {
            "ok": True,
            "tasks": store.list_tasks(
                status=_optional_text(args.get("status")),
                project_id=_optional_text(args.get("project_id")),
                due_before=_optional_text(args.get("due_before")),
                include_completed=bool(args.get("include_completed", False)),
                query=_optional_text(args.get("query")),
                limit=_positive_int(args.get("limit"), default=10),
            ),
        },
    )


def _tasks_complete(store: OrganizerStore) -> AgentTool:
    return AgentTool(
        name="tasks_complete",
        description="Mark a task complete by id or matching title text.",
        parameters=object_schema(
            {
                "task_id": nullable_string_schema("Exact task id if known."),
                "title_query": nullable_string_schema("Task title search text."),
            },
        ),
        handler=lambda args: {
            "ok": True,
            "task": store.complete_task(
                task_id=_optional_text(args.get("task_id")),
                title_query=_optional_text(args.get("title_query")),
            ),
        },
    )


def _projects_create(store: OrganizerStore) -> AgentTool:
    return AgentTool(
        name="projects_create",
        description="Create a project or goal bucket for related notes, tasks, and decisions.",
        parameters=object_schema(
            {
                "name": string_schema("Project or goal name."),
                "description": nullable_string_schema("Optional project description."),
                "status": nullable_string_schema("Status such as active, paused, or done."),
                "tags": string_array_schema("Optional lowercase tags."),
            },
            required=["name"],
        ),
        handler=lambda args: {
            "ok": True,
            "project": store.create_project(
                name=str(args["name"]),
                description=_optional_text(args.get("description")),
                status=_optional_text(args.get("status")),
                tags=_string_list(args.get("tags")),
            ),
        },
    )


def _projects_list(store: OrganizerStore) -> AgentTool:
    return AgentTool(
        name="projects_list",
        description="List projects or goals by status or search text.",
        parameters=object_schema(
            {
                "status": nullable_string_schema("Optional status filter."),
                "query": nullable_string_schema("Optional search text."),
                "limit": integer_schema("Maximum projects to return.", minimum=1),
            },
        ),
        handler=lambda args: {
            "ok": True,
            "projects": store.list_projects(
                status=_optional_text(args.get("status")),
                query=_optional_text(args.get("query")),
                limit=_positive_int(args.get("limit"), default=10),
            ),
        },
    )


def _daily_plan_get(store: OrganizerStore) -> AgentTool:
    return AgentTool(
        name="daily_plan_get",
        description="Build a daily plan view from saved plan, tasks, reminders, and itinerary.",
        parameters=object_schema(
            {
                "date": string_schema("ISO date for the daily plan."),
                "include_completed": boolean_schema("Whether completed items should be included."),
            },
            required=["date"],
        ),
        handler=lambda args: store.get_daily_plan(
            date=str(args["date"]),
            include_completed=bool(args.get("include_completed", False)),
        ),
    )


def _time_sensitive_check(store: OrganizerStore) -> AgentTool:
    return AgentTool(
        name="time_sensitive_check",
        description=(
            "Check up-next, next-up, important, or otherwise timely items: "
            "due reminders, due tasks, and itinerary entries."
        ),
        parameters=object_schema(
            {
                "date_from": nullable_string_schema("ISO start date for the window."),
                "date_to": nullable_string_schema("ISO end date for the window."),
                "days_ahead": integer_schema(
                    "Lookahead days when date_to is not provided.",
                    minimum=0,
                ),
                "include_overdue": boolean_schema(
                    "Whether overdue open tasks/reminders before date_from should be included."
                ),
                "include_completed": boolean_schema(
                    "Whether completed tasks/reminders are included."
                ),
                "important_only": boolean_schema(
                    "Whether to include high-priority open tasks/reminders even without a due date."
                ),
                "limit": integer_schema("Maximum items per category to return.", minimum=1),
            },
        ),
        handler=lambda args: store.list_time_sensitive(
            date_from=_optional_text(args.get("date_from")),
            date_to=_optional_text(args.get("date_to")),
            days_ahead=_nonnegative_int(args.get("days_ahead"), default=7),
            include_overdue=bool(args.get("include_overdue", True)),
            include_completed=bool(args.get("include_completed", False)),
            important_only=bool(args.get("important_only", False)),
            limit=_positive_int(args.get("limit"), default=10),
        ),
    )


def _daily_plan_save(store: OrganizerStore) -> AgentTool:
    return AgentTool(
        name="daily_plan_save",
        description="Persist a curated daily plan for a date.",
        parameters=object_schema(
            {
                "date": string_schema("ISO date for the daily plan."),
                "title": nullable_string_schema("Optional plan title."),
                "focus": nullable_string_schema("Optional main focus."),
                "items": string_array_schema("Plan items in order."),
                "notes": nullable_string_schema("Optional notes."),
            },
            required=["date"],
        ),
        handler=lambda args: {
            "ok": True,
            "daily_plan": store.save_daily_plan(
                date=str(args["date"]),
                title=_optional_text(args.get("title")),
                focus=_optional_text(args.get("focus")),
                items=_string_list(args.get("items")),
                notes=_optional_text(args.get("notes")),
            ),
        },
    )


def _decisions_add(store: OrganizerStore) -> AgentTool:
    return AgentTool(
        name="decisions_add",
        description="Record a decision, rationale, date, and optional project link.",
        parameters=object_schema(
            {
                "title": string_schema("Decision title."),
                "decision": string_schema("What was decided."),
                "rationale": nullable_string_schema("Why this decision was made."),
                "project_id": nullable_string_schema("Project id if related."),
                "decided_on": nullable_string_schema("ISO date if known."),
                "tags": string_array_schema("Optional lowercase tags."),
            },
            required=["title", "decision"],
        ),
        handler=lambda args: {
            "ok": True,
            "decision": store.add_decision(
                title=str(args["title"]),
                decision=str(args["decision"]),
                rationale=_optional_text(args.get("rationale")),
                project_id=_optional_text(args.get("project_id")),
                decided_on=_optional_text(args.get("decided_on")),
                tags=_string_list(args.get("tags")),
            ),
        },
    )


def _decisions_list(store: OrganizerStore) -> AgentTool:
    return AgentTool(
        name="decisions_list",
        description="List decision log entries by project or search text.",
        parameters=object_schema(
            {
                "project_id": nullable_string_schema("Project id filter."),
                "query": nullable_string_schema("Optional search text."),
                "limit": integer_schema("Maximum decisions to return.", minimum=1),
            },
        ),
        handler=lambda args: {
            "ok": True,
            "decisions": store.list_decisions(
                project_id=_optional_text(args.get("project_id")),
                query=_optional_text(args.get("query")),
                limit=_positive_int(args.get("limit"), default=10),
            ),
        },
    )


def _people_add(store: OrganizerStore) -> AgentTool:
    return AgentTool(
        name="people_add",
        description="Save a person/contact note with relationship, organization, and follow-up.",
        parameters=object_schema(
            {
                "name": string_schema("Person name."),
                "relationship": nullable_string_schema("Relationship or role."),
                "organization": nullable_string_schema("Company, team, or group."),
                "notes": nullable_string_schema("Notes about the person."),
                "follow_up_date": nullable_string_schema("ISO follow-up date if known."),
                "tags": string_array_schema("Optional lowercase tags."),
            },
            required=["name"],
        ),
        handler=lambda args: {
            "ok": True,
            "person": store.add_person(
                name=str(args["name"]),
                relationship=_optional_text(args.get("relationship")),
                organization=_optional_text(args.get("organization")),
                notes=_optional_text(args.get("notes")),
                follow_up_date=_optional_text(args.get("follow_up_date")),
                tags=_string_list(args.get("tags")),
            ),
        },
    )


def _people_note_add(store: OrganizerStore) -> AgentTool:
    return AgentTool(
        name="people_note_add",
        description="Append a timestamped note to an existing person/contact record.",
        parameters=object_schema(
            {
                "person_id": nullable_string_schema("Exact person id if known."),
                "name_query": nullable_string_schema("Person name search text."),
                "note": string_schema("Note to append."),
            },
            required=["note"],
        ),
        handler=lambda args: {
            "ok": True,
            "person": store.add_person_note(
                person_id=_optional_text(args.get("person_id")),
                name_query=_optional_text(args.get("name_query")),
                note=str(args["note"]),
            ),
        },
    )


def _people_list(store: OrganizerStore) -> AgentTool:
    return AgentTool(
        name="people_list",
        description="List people/contact notes by name, organization, role, or note text.",
        parameters=object_schema(
            {
                "query": nullable_string_schema("Optional search text."),
                "limit": integer_schema("Maximum people to return.", minimum=1),
            },
        ),
        handler=lambda args: {
            "ok": True,
            "people": store.list_people(
                query=_optional_text(args.get("query")),
                limit=_positive_int(args.get("limit"), default=10),
            ),
        },
    )


def _organization_summary(store: OrganizerStore) -> AgentTool:
    return AgentTool(
        name="organization_summary",
        description="Return counts for all organizer categories and open loops.",
        parameters=object_schema({}),
        handler=lambda _args: store.summary(),
    )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _nonnegative_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, parsed)
