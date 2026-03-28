import pygame
import simpn.visualisation as vis

from simpn.simulator import SimVar, SimVarQueue, SimEvent

TASK_TOKEN_SHOW_COLOURS = [
    pygame.Color("#2E86AB"),  # Deep blue
    pygame.Color("#A23B72"),  # Magenta
    pygame.Color("#F18F01"),  # Orange
    pygame.Color("#C73E1D"),  # Red
    pygame.Color("#6A994E"),  # Green
    pygame.Color("#7209B7"),  # Purple
    pygame.Color("#F77F00"),  # Dark orange
    pygame.Color("#06A77D"),  # Teal
    pygame.Color("#6C757D"),  # Gray
]


class OCPNVar(SimVar):
    """
    An OCPNVar represents a typed place in an object-centric Petri net.
    Each variable corresponds to an object type and holds tokens representing objects of that type.

    Attributes:
        object_type: The type of objects this variable holds
    """

    # Class-level color dictionary shared across all instances
    _shared_object_type_colors = {}
    _color_counter = 0  # Counter for sequential color assignment

    def __init__(self, model, object_type, _id, priority=None):
        """
        Initialize an OCPNVar.

        Args:
            model: The SimProblem model this variable belongs to
            object_type: The type of objects this variable holds (e.g., "PO", "item")
            _id: The identifier/name of this variable
            priority: Optional priority function for token ordering
        """
        super().__init__(_id, priority)
        self.object_type = object_type
        # Register this variable with the model using add_prototype_var
        model.add_prototype_var(self)
        # Create a queue for variable arcs (after super init sets _id)
        self.queue = SimVarQueue(self)

    def put(self, *tokens):
        """
        Put tokens (objects) into this variable.
        Tokens should be dictionaries with an 'object_type' field matching this variable's type.

        Args:
            *tokens: Variable number of token dictionaries to put into this variable
        """
        # Validate that tokens have the correct object_type (optional validation)
        for token in tokens:
            if isinstance(token, dict) and "object_type" in token:
                if token["object_type"] != self.object_type:
                    raise ValueError(
                        f"Token object_type '{token['object_type']}' does not match variable type '{self.object_type}'"
                    )
            elif isinstance(token, dict) and "object_type" not in token:
                raise ValueError(
                    f"Token '{token}' is a dictionary but does not have an 'object_type' field"
                )
            elif not isinstance(token, dict):
                raise ValueError(f"Token '{token}' is not a dictionary")

        super().put(*tokens)

    class OCPNVarViz(vis.PlaceViz):
        def __init__(self, model_node):
            super().__init__(model_node)
            # Get object_type from the model_node (which is the OCPNVar instance)
            self._object_type = None

            if hasattr(model_node, "object_type"):
                self._object_type = model_node.object_type

        def get_color_for_object_type(self, object_type):
            """Get a consistent color for an object type (shared across all instances)."""
            if object_type not in OCPNVar._shared_object_type_colors:
                # Use sequential assignment to ensure different object types get different colors
                # If we run out of colors, cycle through them
                color_idx = OCPNVar._color_counter % len(TASK_TOKEN_SHOW_COLOURS)
                OCPNVar._shared_object_type_colors[object_type] = (
                    TASK_TOKEN_SHOW_COLOURS[color_idx]
                )
                OCPNVar._color_counter += 1
            return OCPNVar._shared_object_type_colors[object_type]

        def draw(self, screen):
            # Use custom color if object_type is set, otherwise use default
            if self._object_type:
                fill_color = self.get_color_for_object_type(self._object_type)
                # Use a darker version for the border
                border_color = pygame.Color(
                    max(0, fill_color.r - 40),
                    max(0, fill_color.g - 40),
                    max(0, fill_color.b - 40),
                )
            else:
                fill_color = vis.TUE_LIGHTBLUE
                border_color = vis.TUE_BLUE

            pygame.draw.circle(
                screen, fill_color, (self._pos[0], self._pos[1]), self._half_height
            )
            pygame.draw.circle(
                screen,
                border_color,
                (self._pos[0], self._pos[1]),
                self._half_height,
                vis.LINE_WIDTH,
            )
            font = pygame.font.SysFont("Calibri", vis.TEXT_SIZE)

            # draw label
            label = font.render(self._model_node.get_id(), True, border_color)
            text_x_pos = self._pos[0] - int(label.get_width() / 2)
            text_y_pos = self._pos[1] + self._half_height + vis.LINE_WIDTH
            screen.blit(label, (text_x_pos, text_y_pos))

            # draw marking as tokens
            vis.TokenShower(self._model_node.marking).set_pos(self._pos).set_time(
                self._curr_time
            ).show_token_count().draw(screen)

    def get_visualisation(self):
        """Return visualization for this typed variable."""
        return self.OCPNVarViz(self)


class OCPNEvent(SimEvent):
    """
    An OCPNEvent represents a transition in an object-centric Petri net.
    It can consume and produce collections of objects from multiple typed places (variables).

    Attributes:
        incoming_vars: List of (OCPNVar, variable_arc [True/False]) instances from which tokens are consumed
        outgoing_vars: List of (OCPNVar, variable_arc [True/False]) instances to which tokens are produced
        guard: guard function that determines when the event can fire
        behavior: behavior function that defines token consumption/production
    """

    def __init__(
        self, model, incoming_vars, outgoing_vars, name, guard=None, behavior=None
    ):
        """
        Initialize an OCPNEvent.

        Args:
            model: The SimProblem model this event belongs to
            incoming_vars: List of (OCPNVar, variable_arc [True/False]) instances from which tokens are consumed
            outgoing_vars: List of (OCPNVar, variable_arc [True/False]) instances to which tokens are produced
            name: The name of this event
            guard: Optional guard function that returns a boolean indicating if the event can fire (default: None)
            behavior: Required behavior function that returns a list of tokens to be put into the outgoing variables
        """
        # Extract the underlying SimVar instances from OCPNVar objects
        incoming_simvars = []
        for var, is_variable in incoming_vars:
            if isinstance(var, OCPNVar):
                if is_variable:
                    incoming_simvars.append(var.queue)
                else:
                    incoming_simvars.append(var)
            else:
                incoming_simvars.append(var)

        outgoing_simvars = []
        for var, is_variable in outgoing_vars:
            if isinstance(var, OCPNVar):
                if is_variable:
                    outgoing_simvars.append(var.queue)
                else:
                    outgoing_simvars.append(var)
            else:
                outgoing_simvars.append(var)

        # Ensure behavior for returning objects not consumed in (p,t) with variable arcs
        return_variable_arcs = []
        for var, is_variable in incoming_vars:
            if isinstance(var, OCPNVar) and is_variable:
                return_variable_arcs.append(var.queue)

        outgoing_simvars.extend(return_variable_arcs)

        # Store the original OCPNVar references before creating wrappers
        self.incoming_vars = incoming_vars
        self.outgoing_vars = outgoing_vars
        self._user_guard = guard
        self._user_behavior = behavior

        # Create wrapped guard and behavior functions with object type and variable arc return checking
        wrapped_guard = self._create_wrapped_guard(guard, incoming_vars)
        wrapped_behavior = self._create_wrapped_behavior(
            behavior, incoming_vars, outgoing_vars, return_variable_arcs
        )

        # Initialize SimEvent with the extracted SimVars and wrapped functions
        super().__init__(
            _id=name,
            guard=wrapped_guard,
            behavior=wrapped_behavior,
            incoming=incoming_simvars,
            outgoing=outgoing_simvars,
        )

        # Add this event to the model using add_prototype_event
        model.add_prototype_event(self)

    def _create_wrapped_guard(self, user_guard, incoming_vars):
        """Create a guard function that wraps the user's guard with object type checking."""

        def wrapped_guard(*args):
            # Check object type constraints for incoming tokens
            for i, arg in enumerate(args):
                if i < len(incoming_vars):
                    var, is_variable = incoming_vars[i]
                    if isinstance(var, OCPNVar):
                        expected_object_type = var.object_type

                        if is_variable:
                            # For variable arcs, arg is a queue
                            if hasattr(arg, "__iter__"):
                                for token in arg:
                                    if hasattr(token, "value") and isinstance(
                                        token.value, dict
                                    ):
                                        token_obj_type = token.value.get("object_type")
                                        if (
                                            token_obj_type
                                            and token_obj_type != expected_object_type
                                        ):
                                            # Guard fails if token type doesn't match
                                            return False
                        else:
                            # For regular arcs, arg is a single token value
                            if isinstance(arg, dict):
                                token_obj_type = arg.get("object_type")
                                if (
                                    token_obj_type
                                    and token_obj_type != expected_object_type
                                ):
                                    # Guard fails if token type doesn't match
                                    return False

            # Call user's guard function if provided
            if user_guard:
                return user_guard(*args)
            return True

        return wrapped_guard

    def _create_wrapped_behavior(
        self, user_behavior, incoming_vars, outgoing_vars, return_variable_arcs
    ):
        """Create a behavior function that wraps the user's behavior with object type checking."""

        def wrapped_behavior(*args):
            # Call user's behavior function
            if user_behavior:
                result = user_behavior(*args)

            # Map outgoing vars to their object types
            # Note: return_variable_arcs are appended at the end of outgoing_vars
            num_return_arcs = len(return_variable_arcs)
            num_outgoing = len(outgoing_vars)

            # Check that returned tokens match their destination object types
            for i, output in enumerate(result):
                # Determine which outgoing var this corresponds to
                if i < num_outgoing:
                    var, is_variable = outgoing_vars[i]
                    if isinstance(var, OCPNVar):
                        expected_object_type = var.object_type

                        if is_variable:
                            # For variable arcs, output is a list/queue of tokens
                            if hasattr(output, "__iter__") and not isinstance(
                                output, str
                            ):
                                for token in output:
                                    if hasattr(token, "value") and isinstance(
                                        token.value, dict
                                    ):
                                        token_obj_type = token.value.get("object_type")
                                        if (
                                            token_obj_type
                                            and token_obj_type != expected_object_type
                                        ):
                                            raise ValueError(
                                                f"Behavior error: Token with object_type '{token_obj_type}' "
                                                f"cannot be returned to place '{var.get_id()}' which expects object_type '{expected_object_type}'"
                                            )
                        else:
                            # For regular arcs, output is a single token
                            if hasattr(output, "value") and isinstance(
                                output.value, dict
                            ):
                                token_obj_type = output.value.get("object_type")
                                if (
                                    token_obj_type
                                    and token_obj_type != expected_object_type
                                ):
                                    raise ValueError(
                                        f"Behavior error: Token with object_type '{token_obj_type}' "
                                        f"cannot be returned to place '{var.get_id()}' which expects object_type '{expected_object_type}'"
                                    )
                            elif isinstance(output, dict):
                                token_obj_type = output.get("object_type")
                                if (
                                    token_obj_type
                                    and token_obj_type != expected_object_type
                                ):
                                    raise ValueError(
                                        f"Behavior error: Token with object_type '{token_obj_type}' "
                                        f"cannot be returned to place '{var.get_id()}' which expects object_type '{expected_object_type}'"
                                    )
                elif i < num_outgoing + num_return_arcs:
                    # This is a return variable arc - find the corresponding incoming var
                    return_idx = i - num_outgoing
                    if return_idx < len(incoming_vars):
                        var, is_variable = incoming_vars[return_idx]
                        if isinstance(var, OCPNVar) and is_variable:
                            expected_object_type = var.object_type
                            # Check tokens in return queue
                            if hasattr(output, "__iter__") and not isinstance(
                                output, str
                            ):
                                for token in output:
                                    if hasattr(token, "value") and isinstance(
                                        token.value, dict
                                    ):
                                        token_obj_type = token.value.get("object_type")
                                        if (
                                            token_obj_type
                                            and token_obj_type != expected_object_type
                                        ):
                                            raise ValueError(
                                                f"Behavior error: Token with object_type '{token_obj_type}' "
                                                f"cannot be returned to place '{var.get_id()}' which expects object_type '{expected_object_type}'"
                                            )

            return result

        return wrapped_behavior

    class OCPNEventViz(vis.TransitionViz):
        def __init__(self, model_node):
            super().__init__(model_node)
            # Get incoming object types from the model_node (which is the OCPNEvent instance)
            self._incoming_object_types_list = []  # List of object types in order of incoming vars

            if hasattr(model_node, "incoming_vars"):
                # Store object types for each incoming var (in order)
                for var, is_variable in model_node.incoming_vars:
                    if isinstance(var, OCPNVar) and hasattr(var, "object_type"):
                        self._incoming_object_types_list.append(var.object_type)
                    else:
                        # For non-OCPNVar vars, use None to represent default/unknown
                        self._incoming_object_types_list.append(None)

        def get_color_for_object_type(self, object_type):
            """Get a consistent color for an object type (same as OCPNVarViz, using shared dict)."""
            if object_type is None:
                return vis.TUE_LIGHTBLUE

            # Use the shared color dictionary from OCPNVar
            if object_type not in OCPNVar._shared_object_type_colors:
                # Use sequential assignment to ensure different object types get different colors
                # If we run out of colors, cycle through them
                color_idx = OCPNVar._color_counter % len(TASK_TOKEN_SHOW_COLOURS)
                OCPNVar._shared_object_type_colors[object_type] = (
                    TASK_TOKEN_SHOW_COLOURS[color_idx]
                )
                OCPNVar._color_counter += 1
            return OCPNVar._shared_object_type_colors[object_type]

        def draw(self, screen):
            # Calculate rectangle bounds
            rect_left = self._pos[0] - self._half_width
            rect_top = self._pos[1] - self._half_height
            rect_width = self._width
            rect_height = self._height

            # Draw proportional sections if we have incoming object types
            if self._incoming_object_types_list:
                total_vars = len(self._incoming_object_types_list)
                section_width = rect_width / total_vars
                gradient_width = min(
                    section_width * 0.3, 10
                )  # 30% of section width or max 10 pixels

                # Draw each section with gradient blending
                for i, object_type in enumerate(self._incoming_object_types_list):
                    section_left = rect_left + i * section_width
                    fill_color = self.get_color_for_object_type(object_type)

                    # Determine next color for gradient (if not last section)
                    if i < len(self._incoming_object_types_list) - 1:
                        next_object_type = self._incoming_object_types_list[i + 1]
                        next_color = self.get_color_for_object_type(next_object_type)
                    else:
                        next_color = None

                    # Draw main section
                    main_section_width = section_width - (
                        gradient_width if next_color else 0
                    )
                    pygame.draw.rect(
                        screen,
                        fill_color,
                        pygame.Rect(
                            section_left,
                            rect_top,
                            main_section_width,
                            rect_height,
                        ),
                    )

                    # Draw gradient transition to next section
                    if next_color:
                        gradient_start = section_left + main_section_width
                        gradient_steps = int(gradient_width)
                        for step in range(gradient_steps):
                            # Interpolate between current and next color
                            ratio = step / gradient_steps
                            blended_color = pygame.Color(
                                int(
                                    fill_color.r + (next_color.r - fill_color.r) * ratio
                                ),
                                int(
                                    fill_color.g + (next_color.g - fill_color.g) * ratio
                                ),
                                int(
                                    fill_color.b + (next_color.b - fill_color.b) * ratio
                                ),
                            )
                            step_x = gradient_start + (
                                step * gradient_width / gradient_steps
                            )
                            step_width = gradient_width / gradient_steps
                            pygame.draw.rect(
                                screen,
                                blended_color,
                                pygame.Rect(
                                    step_x,
                                    rect_top,
                                    step_width,
                                    rect_height,
                                ),
                            )
            else:
                # Default color if no incoming object types
                fill_color = vis.TUE_LIGHTBLUE
                pygame.draw.rect(
                    screen,
                    fill_color,
                    pygame.Rect(
                        rect_left,
                        rect_top,
                        rect_width,
                        rect_height,
                    ),
                )

            # Draw border
            border_color = vis.TUE_BLUE
            pygame.draw.rect(
                screen,
                border_color,
                pygame.Rect(
                    rect_left,
                    rect_top,
                    rect_width,
                    rect_height,
                ),
                vis.LINE_WIDTH,
            )

            font = pygame.font.SysFont("Calibri", vis.TEXT_SIZE)

            # draw label
            label = font.render(self._model_node.get_id(), True, border_color)
            text_x_pos = self._pos[0] - int(label.get_width() / 2)
            text_y_pos = self._pos[1] + self._half_height + vis.LINE_WIDTH
            screen.blit(label, (text_x_pos, text_y_pos))

    def get_visualisation(self):
        """Return visualization for this event."""
        return self.OCPNEventViz(self)
