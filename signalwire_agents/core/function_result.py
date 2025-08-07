"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""
SwaigFunctionResult class for handling the response format of SWAIG function calls
"""

from typing import Dict, List, Any, Optional, Union


class SwaigFunctionResult:
    """
    Wrapper around SWAIG function responses that handles proper formatting
    of response text and actions.
    
    The result object has three main components:
    1. response: Text the AI should say back to the user
    2. action: List of structured actions to execute 
    3. post_process: Whether to let AI take another turn before executing actions
    
    Post-processing behavior:
    - post_process=False (default): Execute actions immediately after AI response
    - post_process=True: Let AI respond to user one more time, then execute actions
    
    This is useful for confirmation workflows like:
    "I'll transfer you to sales. Do you have any other questions first?"
    (AI can handle follow-up, then execute the transfer)
    
    Example:
        return SwaigFunctionResult("Found your order")
        
        # With actions
        return (
            SwaigFunctionResult("I'll transfer you to support")
            .add_action("transfer", {"dest": "support"})
        )
        
        # With simple action value
        return (
            SwaigFunctionResult("I'll confirm that")
            .add_action("confirm", True)
        )
        
        # With multiple actions
        return (
            SwaigFunctionResult("Processing your request")
            .add_actions([
                {"set_global_data": {"key": "value"}},
                {"play": {"url": "music.mp3"}}
            ])
        )
        
        # With post-processing enabled
        return (
            SwaigFunctionResult("Let me transfer you to billing", post_process=True)
            .connect("+15551234567", final=True)
        )
        
        # Using the connect helper
        return (
            SwaigFunctionResult("I'll transfer you to our sales team now")
            .connect("sales@company.com", final=False, from_addr="+15559876543")
        )
    """
    def __init__(self, response: Optional[str] = None, post_process: bool = False):
        """
        Initialize a new SWAIG function result
        
        Args:
            response: Optional natural language response to include
            post_process: Whether to let AI take another turn before executing actions.
                         Defaults to False (execute actions immediately after response).
        """
        self.response = response or ""
        self.action: List[Dict[str, Any]] = []
        self.post_process = post_process
    
    def set_response(self, response: str) -> 'SwaigFunctionResult':
        """
        Set the natural language response text
        
        Args:
            response: The text the AI should say
            
        Returns:
            Self for method chaining
        """
        self.response = response
        return self
    
    def set_post_process(self, post_process: bool) -> 'SwaigFunctionResult':
        """
        Set whether to enable post-processing for this result.
        
        Post-processing allows the AI to take one more turn with the user
        before executing any actions. This is useful for confirmation workflows.
        
        Args:
            post_process: True to let AI respond once more before executing actions,
                         False to execute actions immediately after the response.
                         
        Returns:
            Self for method chaining
        """
        self.post_process = post_process
        return self
    
    def add_action(self, name: str, data: Any) -> 'SwaigFunctionResult':
        """
        Add a structured action to the response
        
        Args:
            name: The name/type of the action (e.g., "play", "transfer")
            data: The data for the action - can be a string, boolean, object, or array
            
        Returns:
            Self for method chaining
        """
        self.action.append({name: data})
        return self
    
    def add_actions(self, actions: List[Dict[str, Any]]) -> 'SwaigFunctionResult':
        """
        Add multiple structured actions to the response
        
        Args:
            actions: List of action objects to add to the response
            
        Returns:
            Self for method chaining
        """
        self.action.extend(actions)
        return self
    
    def connect(self, destination: str, final: bool = True, from_addr: Optional[str] = None) -> 'SwaigFunctionResult':
        """
        Add a connect action to transfer/connect the call to another destination.
        
        This is a convenience method that abstracts the SWML connect verb, so users
        don't need to manually construct SWML documents.
        
        Transfer behavior:
        - final=True: Permanent transfer - call exits the agent completely, 
                     SWML replaces the agent and call continues there
        - final=False: Temporary transfer - if far end hangs up, call returns 
                      to the agent to continue the conversation
        
        Args:
            destination: Where to connect the call (phone number, SIP address, etc.)
            final: Whether this is a permanent transfer (True) or temporary (False).
                  Defaults to True for permanent transfers.
            from_addr: Optional caller ID override (phone number or SIP address).
                      If None, uses the current call's from address.
                      
        Returns:
            Self for method chaining
            
        Example:
            # Permanent transfer to a phone number
            result.connect("+15551234567", final=True)
            
            # Temporary transfer to SIP address with custom caller ID
            result.connect("support@company.com", final=False, from_addr="+15559876543")
        """
        # Build the connect verb parameters
        connect_params = {"to": destination}
        if from_addr is not None:
            connect_params["from"] = from_addr
        
        # Create the SWML action
        swml_action = {
            "SWML": {
                "sections": {
                    "main": [{"connect": connect_params}]
                },
                "version": "1.0.0"
            },
            "transfer": str(final).lower()  # Convert boolean to "true"/"false" string
        }
        
        # Add to actions list
        self.action.append(swml_action)
        return self

    def swml_transfer(self, dest: str, ai_response: str, final: bool = True) -> 'SwaigFunctionResult':
        """
        Add a SWML transfer action with AI response setup for when transfer completes.
        
        This is a virtual helper that generates SWML to transfer the call to another
        destination and sets up an AI response for when the transfer completes and
        control returns to the agent.
        
        For transfers, you typically want to enable post-processing so the AI speaks
        the response first before executing the transfer.
        
        Args:
            dest: Destination URL for the transfer (SWML endpoint, SIP address, etc.)
            ai_response: Message the AI should say when transfer completes and control returns
            final: Whether this is a permanent transfer (True) or temporary (False).
                  Defaults to True for permanent transfers (same as connect method).
                        
        Returns:
            Self for method chaining
            
        Example:
            # Permanent transfer (default)
            result = (
                SwaigFunctionResult("I'm transferring you to support", post_process=True)
                .swml_transfer(
                    "https://support.example.com/swml",
                    "Goodbye!"  # Won't be used since final=True by default
                )
            )
            
            # Temporary transfer with return
            result.swml_transfer(
                dest, 
                "The support call is complete. How else can I help?", 
                final=False
            )
        """
        # Create the SWML action structure directly
        swml_action = {
            "SWML": {
                "version": "1.0.0",
                "sections": {
                    "main": [
                        {"set": {"ai_response": ai_response}},
                        {"transfer": {"dest": dest}}
                    ]
                }
            },
            "transfer": str(final).lower()  # Convert boolean to "true"/"false" string
        }
        
        # Add to actions list directly
        self.action.append(swml_action)
        
        return self
    
    def update_global_data(self, data: Dict[str, Any]) -> 'SwaigFunctionResult':
        """
        Update global agent data variables.
        
        This is a convenience method that abstracts the set_global_data action.
        Global data persists across the entire agent session and is available
        in prompt variables and can be accessed by all functions.
        
        Args:
            data: Dictionary of key-value pairs to set/update in global data
            
        Returns:
            self for method chaining
        """
        return self.add_action("set_global_data", data)

    def swml_user_event(self, event_data: Dict[str, Any]) -> 'SwaigFunctionResult':
        """
        Send a user event through SWML to update the client UI.
        
        This is a convenience method for sending user events to connected clients,
        commonly used for real-time UI updates in interactive applications.
        
        Args:
            event_data: Dictionary containing the event type and any associated data
                       Example: {"type": "cards_dealt", "player_hand": [...], "score": 21}
            
        Returns:
            Self for method chaining
            
        Example:
            result = (
                SwaigFunctionResult("You have blackjack!")
                .swml_user_event({
                    "type": "cards_dealt",
                    "player_hand": player_cards,
                    "dealer_hand": dealer_cards,
                    "player_score": 21
                })
            )
        """
        swml_action = {
            "sections": {
                "main": [{
                    "user_event": {
                        "event": event_data
                    }
                }]
            },
            "version": "1.0.0"
        }
        
        return self.add_action("SWML", swml_action)
    
    def swml_change_step(self, step_name: str) -> 'SwaigFunctionResult':
        """
        Change the conversation step in the AI agent's workflow.
        
        This is a convenience method for transitioning between conversation steps,
        allowing dynamic workflow control based on user interactions or game state.
        
        Args:
            step_name: Name of the step to transition to (e.g., "betting", "playing", "hand_complete")
            
        Returns:
            Self for method chaining
            
        Example:
            result = (
                SwaigFunctionResult("Starting a new hand")
                .swml_change_step("betting")
                .swml_user_event({"type": "game_reset", "chips": 1000})
            )
        """
        return self.add_action("change_step", step_name)
    
    def swml_change_context(self, context_name: str) -> 'SwaigFunctionResult':
        """
        Change the conversation context in the AI agent's workflow.
        
        This is a convenience method for switching between different conversation contexts,
        useful for agents that handle multiple distinct workflows or service modes.
        
        Args:
            context_name: Name of the context to transition to (e.g., "support", "sales", "technical")
            
        Returns:
            Self for method chaining
            
        Example:
            result = (
                SwaigFunctionResult("Transferring you to technical support")
                .swml_change_context("technical_support")
            )
        """
        return self.add_action("change_context", context_name)
    
    def execute_swml(self, swml_content, transfer: bool = False) -> 'SwaigFunctionResult':
        """
        Execute SWML content with optional transfer behavior.
        
        Args:
            swml_content: Can be:
                - String: Raw SWML JSON text
                - Dict: SWML data structure
                - SWML object: SignalWire SWML SDK object with .to_dict() method
            transfer: Boolean - whether call should exit agent after execution
            
        Returns:
            self for method chaining
        """
        # Detect input type and normalize to appropriate format
        if isinstance(swml_content, str):
            # Raw SWML string - use as-is
            swml_data = swml_content
        elif hasattr(swml_content, 'to_dict'):
            # SWML SDK object - convert to dict
            swml_data = swml_content.to_dict()
        elif isinstance(swml_content, dict):
            # Dict - use directly
            swml_data = swml_content
        else:
            raise TypeError("swml_content must be string, dict, or SWML object")
        
        action = swml_data
        if transfer:
            action["transfer"] = "true"
        
        return self.add_action("SWML", action)

    def hangup(self) -> 'SwaigFunctionResult':
        """
        Terminate the call.
        
        Returns:
            self for method chaining
        """
        return self.add_action("hangup", True)

    def hold(self, timeout: int = 300) -> 'SwaigFunctionResult':
        """
        Put the call on hold with optional timeout.
        
        Args:
            timeout: Timeout in seconds (max 900, default 300)
            
        Returns:
            self for method chaining
        """
        # Clamp timeout to valid range
        timeout = max(0, min(timeout, 900))
        return self.add_action("hold", timeout)

    def wait_for_user(self, enabled: Optional[bool] = None, timeout: Optional[int] = None, answer_first: bool = False) -> 'SwaigFunctionResult':
        """
        Control how agent waits for user input.
        
        Args:
            enabled: Boolean to enable/disable waiting
            timeout: Number of seconds to wait
            answer_first: Special "answer_first" mode
            
        Returns:
            self for method chaining
        """
        if answer_first:
            wait_value = "answer_first"
        elif timeout is not None:
            wait_value = timeout
        elif enabled is not None:
            wait_value = enabled
        else:
            wait_value = True
            
        return self.add_action("wait_for_user", wait_value)

    def stop(self) -> 'SwaigFunctionResult':
        """
        Stop the agent execution.
        
        Returns:
            self for method chaining
        """
        return self.add_action("stop", True)

    def say(self, text: str) -> 'SwaigFunctionResult':
        """
        Make the agent speak specific text.
        
        Args:
            text: Text for agent to speak
            
        Returns:
            self for method chaining
        """
        return self.add_action("say", text)

    def play_background_file(self, filename: str, wait: bool = False) -> 'SwaigFunctionResult':
        """
        Play audio or video file in background.
        
        Args:
            filename: Audio/video filename/path
            wait: Whether to suppress attention-getting behavior during playback
            
        Returns:
            self for method chaining
        """
        if wait:
            return self.add_action("playback_bg", {"file": filename, "wait": True})
        else:
            return self.add_action("playback_bg", filename)

    def stop_background_file(self) -> 'SwaigFunctionResult':
        """
        Stop currently playing background file.
        
        Returns:
            self for method chaining
        """
        return self.add_action("stop_playback_bg", True)

    def set_end_of_speech_timeout(self, milliseconds: int) -> 'SwaigFunctionResult':
        """
        Adjust end of speech timeout - milliseconds of silence after speaking 
        has been detected to finalize speech recognition.
        
        Args:
            milliseconds: Timeout in milliseconds
            
        Returns:
            self for method chaining
        """
        return self.add_action("end_of_speech_timeout", milliseconds)

    def set_speech_event_timeout(self, milliseconds: int) -> 'SwaigFunctionResult':
        """
        Adjust speech event timeout - milliseconds since last speech detection 
        event to finalize recognition. Works better in noisy environments.
        
        Args:
            milliseconds: Timeout in milliseconds
            
        Returns:
            self for method chaining
        """
        return self.add_action("speech_event_timeout", milliseconds)

    def remove_global_data(self, keys: Union[str, List[str]]) -> 'SwaigFunctionResult':
        """
        Remove global agent data variables.
        
        Args:
            keys: Single key string or list of keys to remove
            
        Returns:
            self for method chaining
        """
        return self.add_action("unset_global_data", keys)

    def set_metadata(self, data: Dict[str, Any]) -> 'SwaigFunctionResult':
        """
        Set metadata scoped to current function's meta_data_token.
        
        Args:
            data: Dictionary of key-value pairs for metadata
            
        Returns:
            self for method chaining
        """
        return self.add_action("set_meta_data", data)

    def remove_metadata(self, keys: Union[str, List[str]]) -> 'SwaigFunctionResult':
        """
        Remove metadata from current function's meta_data_token scope.
        
        Args:
            keys: Single key string or list of keys to remove
            
        Returns:
            self for method chaining
        """
        return self.add_action("unset_meta_data", keys)

    def toggle_functions(self, function_toggles: List[Dict[str, Any]]) -> 'SwaigFunctionResult':
        """
        Enable/disable specific SWAIG functions.
        
        Args:
            function_toggles: List of dicts with 'function' and 'active' keys
            
        Returns:
            self for method chaining
        """
        return self.add_action("toggle_functions", function_toggles)

    def enable_functions_on_timeout(self, enabled: bool = True) -> 'SwaigFunctionResult':
        """
        Enable function calls on speaker timeout.
        
        Args:
            enabled: Whether to enable functions on timeout
            
        Returns:
            self for method chaining
        """
        return self.add_action("functions_on_speaker_timeout", enabled)

    def enable_extensive_data(self, enabled: bool = True) -> 'SwaigFunctionResult':
        """
        Send full data to LLM for this turn only, then use smaller replacement 
        in subsequent turns.
        
        Args:
            enabled: Whether to send extensive data this turn only
            
        Returns:
            self for method chaining
        """
        return self.add_action("extensive_data", enabled)

    def update_settings(self, settings: Dict[str, Any]) -> 'SwaigFunctionResult':
        """
        Update agent runtime settings.
        
        Supported settings:
        - frequency-penalty: Float (-2.0 to 2.0)
        - presence-penalty: Float (-2.0 to 2.0) 
        - max-tokens: Integer (0 to 4096)
        - top-p: Float (0.0 to 1.0)
        - confidence: Float (0.0 to 1.0)
        - barge-confidence: Float (0.0 to 1.0)
        - temperature: Float (0.0 to 2.0, clamped to 1.5)
        
        Args:
            settings: Dictionary of settings to update
            
        Returns:
            self for method chaining
        """
        return self.add_action("settings", settings)

    def switch_context(self, system_prompt: Optional[str] = None, user_prompt: Optional[str] = None, 
                      consolidate: bool = False, full_reset: bool = False) -> 'SwaigFunctionResult':
        """
        Change agent context/prompt during conversation.
        
        Args:
            system_prompt: New system prompt
            user_prompt: User message to add
            consolidate: Whether to summarize existing conversation
            full_reset: Whether to do complete context reset
            
        Returns:
            self for method chaining
        """
        if system_prompt and not user_prompt and not consolidate and not full_reset:
            # Simple string context switch
            return self.add_action("context_switch", system_prompt)
        else:
            # Advanced object context switch
            context_data = {}
            if system_prompt:
                context_data["system_prompt"] = system_prompt
            if user_prompt:
                context_data["user_prompt"] = user_prompt
            if consolidate:
                context_data["consolidate"] = True
            if full_reset:
                context_data["full_reset"] = True
            return self.add_action("context_switch", context_data)

    def simulate_user_input(self, text: str) -> 'SwaigFunctionResult':
        """
        Queue simulated user input.
        
        Args:
            text: Text to simulate as user input
            
        Returns:
            self for method chaining
        """
        return self.add_action("user_input", text)

    def send_sms(self, to_number: str, from_number: str, body: Optional[str] = None, 
                media: Optional[List[str]] = None, tags: Optional[List[str]] = None, 
                region: Optional[str] = None) -> 'SwaigFunctionResult':
        """
        Send a text message to a PSTN phone number using SWML.
        
        This is a virtual helper that generates SWML to send SMS messages.
        Either body or media (or both) must be provided.
        
        Args:
            to_number: Phone number in E.164 format to send to
            from_number: Phone number in E.164 format to send from  
            body: Body text of the message (optional if media provided)
            media: Array of URLs to send in the message (optional if body provided)
            tags: Array of tags to associate with the message for UI searching
            region: Region to originate the message from
            
        Returns:
            self for method chaining
            
        Raises:
            ValueError: If neither body nor media is provided
        """
        # Validate that at least body or media is provided
        if not body and not media:
            raise ValueError("Either body or media must be provided")
        
        # Build the send_sms parameters
        sms_params = {
            "to_number": to_number,
            "from_number": from_number
        }
        
        # Add optional parameters
        if body:
            sms_params["body"] = body
        if media:
            sms_params["media"] = media
        if tags:
            sms_params["tags"] = tags
        if region:
            sms_params["region"] = region
        
        # Generate SWML document
        swml_doc = {
            "version": "1.0.0",
            "sections": {
                "main": [
                    {"send_sms": sms_params}
                ]
            }
        }
        
        # Use execute_swml to add the action
        return self.execute_swml(swml_doc)

    def pay(self, payment_connector_url: str, input_method: str = "dtmf", 
           status_url: Optional[str] = None, payment_method: str = "credit-card",
           timeout: int = 5, max_attempts: int = 1, security_code: bool = True,
           postal_code: Union[bool, str] = True, min_postal_code_length: int = 0,
           token_type: str = "reusable", charge_amount: Optional[str] = None,
           currency: str = "usd", language: str = "en-US", voice: str = "woman",
           description: Optional[str] = None, valid_card_types: str = "visa mastercard amex",
           parameters: Optional[List[Dict[str, str]]] = None,
           prompts: Optional[List[Dict[str, Any]]] = None,
           ai_response: Optional[str] = "The payment status is ${pay_result}, do not mention anything else about collecting payment if successful.") -> 'SwaigFunctionResult':
        """
        Process payment using SWML pay action.
        
        This is a virtual helper that generates SWML for payment processing.
        
        Args:
            payment_connector_url: URL to make payment requests to (required)
            input_method: Method to collect payment details ("dtmf" or "voice")
            status_url: URL for status change notifications
            payment_method: Payment method ("credit-card" currently supported)
            timeout: Seconds to wait for next digit (default: 5)
            max_attempts: Number of retry attempts (default: 1)
            security_code: Whether to prompt for security code (default: True)
            postal_code: Whether to prompt for postal code, or actual postcode
            min_postal_code_length: Minimum postal code digits (default: 0)
            token_type: Payment type ("one-time" or "reusable", default: "reusable")
            charge_amount: Amount to charge as decimal string
            currency: Currency code (default: "usd")
            language: Language for prompts (default: "en-US")
            voice: TTS voice to use (default: "woman")
            description: Custom payment description
            valid_card_types: Space-separated card types (default: "visa mastercard amex")
            parameters: Array of name/value pairs for payment connector
            prompts: Array of custom prompt configurations
            
        Returns:
            self for method chaining
        """
        # Build the pay parameters
        pay_params = {
            "payment_connector_url": payment_connector_url,
            "input": input_method,
            "payment_method": payment_method,
            "timeout": str(timeout),
            "max_attempts": str(max_attempts),
            "security_code": str(security_code).lower(),
            "min_postal_code_length": str(min_postal_code_length),
            "token_type": token_type,
            "currency": currency,
            "language": language,
            "voice": voice,
            "valid_card_types": valid_card_types
        }
        
        # Handle postal_code (can be boolean or string)
        if isinstance(postal_code, bool):
            pay_params["postal_code"] = str(postal_code).lower()
        else:
            pay_params["postal_code"] = postal_code
        
        # Add optional parameters
        if status_url:
            pay_params["status_url"] = status_url
        if charge_amount:
            pay_params["charge_amount"] = charge_amount
        if description:
            pay_params["description"] = description
        if parameters:
            pay_params["parameters"] = parameters
        if prompts:
            pay_params["prompts"] = prompts
        
        # Generate SWML document
        swml_doc = {
            "version": "1.0.0",
            "sections": {
                "main": [
                    {"set": {"ai_response": ai_response}},
                    {"pay": pay_params}
                ]
            }
        }
        
        # Use execute_swml to add the action
        return self.execute_swml(swml_doc)

    def record_call(self, control_id: Optional[str] = None, stereo: bool = False, 
                   format: str = "wav", direction: str = "both", 
                   terminators: Optional[str] = None, beep: bool = False,
                   input_sensitivity: float = 44.0, initial_timeout: float = 0.0,
                   end_silence_timeout: float = 0.0, max_length: Optional[float] = None,
                   status_url: Optional[str] = None) -> 'SwaigFunctionResult':
        """
        Start background call recording using SWML.
        
        This is a virtual helper that generates SWML to start recording the call 
        in the background. Unlike foreground recording, the script continues 
        executing while recording happens in the background.
        
        Args:
            control_id: Identifier for this recording (for use with stop_record_call)
            stereo: Record in stereo (default: False)
            format: Recording format - "wav" or "mp3" (default: "wav") 
            direction: Audio direction - "speak", "listen", or "both" (default: "both")
            terminators: Digits that stop recording when pressed
            beep: Play beep before recording (default: False)
            input_sensitivity: Input sensitivity for recording (default: 44.0)
            initial_timeout: Time in seconds to wait for speech start (default: 0.0)
            end_silence_timeout: Time in seconds to wait in silence before ending (default: 0.0)
            max_length: Maximum recording length in seconds
            status_url: URL to send recording status events to
            
        Returns:
            self for method chaining
        """
        # Validate format parameter
        if format not in ["wav", "mp3"]:
            raise ValueError("format must be 'wav' or 'mp3'")
        
        # Validate direction parameter    
        if direction not in ["speak", "listen", "both"]:
            raise ValueError("direction must be 'speak', 'listen', or 'both'")
        
        # Build the record_call parameters
        record_params = {
            "stereo": stereo,
            "format": format,
            "direction": direction,
            "beep": beep,
            "input_sensitivity": input_sensitivity,
            "initial_timeout": initial_timeout,
            "end_silence_timeout": end_silence_timeout
        }
        
        # Add optional parameters
        if control_id:
            record_params["control_id"] = control_id
        if terminators:
            record_params["terminators"] = terminators
        if max_length:
            record_params["max_length"] = max_length
        if status_url:
            record_params["status_url"] = status_url
        
        # Generate SWML document
        swml_doc = {
            "version": "1.0.0",
            "sections": {
                "main": [
                    {"record_call": record_params}
                ]
            }
        }
        
        # Use execute_swml to add the action
        return self.execute_swml(swml_doc)

    def stop_record_call(self, control_id: Optional[str] = None) -> 'SwaigFunctionResult':
        """
        Stop an active background call recording using SWML.
        
        This is a virtual helper that generates SWML to stop a recording that 
        was started with record_call().
        
        Args:
            control_id: Identifier for the recording to stop. If not provided,
                       the most recent recording will be stopped.
            
        Returns:
            self for method chaining
        """
        # Build the stop_record_call parameters
        stop_params = {}
        if control_id:
            stop_params["control_id"] = control_id
        
        # Generate SWML document
        swml_doc = {
            "version": "1.0.0",
            "sections": {
                "main": [
                    {"stop_record_call": stop_params}
                ]
            }
        }
        
        # Use execute_swml to add the action
        return self.execute_swml(swml_doc)

    def join_room(self, name: str) -> 'SwaigFunctionResult':
        """
        Join a RELAY room using SWML.
        
        This is a virtual helper that generates SWML to join a RELAY room,
        which enables multi-party communication and collaboration.
        
        Args:
            name: The name of the room to join (required)
            
        Returns:
            self for method chaining
        """
        # Build the join_room parameters
        join_params = {"name": name}
        
        # Generate SWML document
        swml_doc = {
            "version": "1.0.0",
            "sections": {
                "main": [
                    {"join_room": join_params}
                ]
            }
        }
        
        # Use execute_swml to add the action
        return self.execute_swml(swml_doc)

    def sip_refer(self, to_uri: str) -> 'SwaigFunctionResult':
        """
        Send SIP REFER to a SIP call using SWML.
        
        This is a virtual helper that generates SWML to send a SIP REFER
        message, which is used for call transfer in SIP environments.
        
        Args:
            to_uri: The SIP URI to send the REFER to (required)
            
        Returns:
            self for method chaining
        """
        # Build the sip_refer parameters
        refer_params = {"to_uri": to_uri}
        
        # Generate SWML document
        swml_doc = {
            "version": "1.0.0",
            "sections": {
                "main": [
                    {"sip_refer": refer_params}
                ]
            }
        }
        
        # Use execute_swml to add the action
        return self.execute_swml(swml_doc)

    def join_conference(self, name: str, muted: bool = False, beep: str = "true", 
                       start_on_enter: bool = True, end_on_exit: bool = False,
                       wait_url: Optional[str] = None, max_participants: int = 250,
                       record: str = "do-not-record", region: Optional[str] = None,
                       trim: str = "trim-silence", coach: Optional[str] = None,
                       status_callback_event: Optional[str] = None, 
                       status_callback: Optional[str] = None,
                       status_callback_method: str = "POST",
                       recording_status_callback: Optional[str] = None,
                       recording_status_callback_method: str = "POST",
                       recording_status_callback_event: str = "completed",
                       result: Optional[Any] = None) -> 'SwaigFunctionResult':
        """
        Join an ad-hoc audio conference with RELAY and CXML calls using SWML.
        
        This is a virtual helper that generates SWML to join audio conferences
        with extensive configuration options for call management and recording.
        
        Args:
            name: Name of conference (required)
            muted: Whether to join muted (default: False)
            beep: Beep configuration - "true", "false", "onEnter", "onExit" (default: "true")
            start_on_enter: Whether conference starts when this participant enters (default: True)
            end_on_exit: Whether conference ends when this participant exits (default: False)
            wait_url: SWML URL for hold music (default: None for default hold music)
            max_participants: Maximum participants <= 250 (default: 250)
            record: Recording mode - "do-not-record", "record-from-start" (default: "do-not-record")
            region: Conference region (default: None)
            trim: Trim silence - "trim-silence", "do-not-trim" (default: "trim-silence")
            coach: SWML Call ID or CXML CallSid for coaching (default: None)
            status_callback_event: Events to report - "start end join leave mute hold modify speaker announcement" (default: None)
            status_callback: URL for status callbacks (default: None)
            status_callback_method: HTTP method - "GET", "POST" (default: "POST")
            recording_status_callback: URL for recording status callbacks (default: None)
            recording_status_callback_method: HTTP method - "GET", "POST" (default: "POST")
            recording_status_callback_event: Recording events - "in-progress completed absent" (default: "completed")
            result: Switch on return_value when object {} or cond when array [] (default: None)
            
        Returns:
            self for method chaining
            
        Raises:
            ValueError: If beep value is invalid or max_participants exceeds 250
        """
        # Validate beep parameter
        valid_beep_values = ["true", "false", "onEnter", "onExit"]
        if beep not in valid_beep_values:
            raise ValueError(f"beep must be one of {valid_beep_values}")
        
        # Validate max_participants
        if max_participants <= 0 or max_participants > 250:
            raise ValueError("max_participants must be a positive integer <= 250")
        
        # Validate record parameter
        valid_record_values = ["do-not-record", "record-from-start"]
        if record not in valid_record_values:
            raise ValueError(f"record must be one of {valid_record_values}")
        
        # Validate trim parameter
        valid_trim_values = ["trim-silence", "do-not-trim"]
        if trim not in valid_trim_values:
            raise ValueError(f"trim must be one of {valid_trim_values}")
        
        # Validate status_callback_method
        valid_methods = ["GET", "POST"]
        if status_callback_method not in valid_methods:
            raise ValueError(f"status_callback_method must be one of {valid_methods}")
        if recording_status_callback_method not in valid_methods:
            raise ValueError(f"recording_status_callback_method must be one of {valid_methods}")
        
        # Build the join_conference parameters - start with required parameter
        if isinstance(name, str) and not name.strip():
            raise ValueError("name cannot be empty")
            
        # For simple case, can just be the conference name
        if (not muted and beep == "true" and start_on_enter and not end_on_exit and 
            wait_url is None and max_participants == 250 and record == "do-not-record" and
            region is None and trim == "trim-silence" and coach is None and
            status_callback_event is None and status_callback is None and
            status_callback_method == "POST" and recording_status_callback is None and
            recording_status_callback_method == "POST" and recording_status_callback_event == "completed" and
            result is None):
            # Simple form - just the conference name
            join_params = name
        else:
            # Full object form with parameters
            join_params = {"name": name}
            
            # Add non-default parameters
            if muted:
                join_params["muted"] = muted
            if beep != "true":
                join_params["beep"] = beep
            if not start_on_enter:
                join_params["start_on_enter"] = start_on_enter
            if end_on_exit:
                join_params["end_on_exit"] = end_on_exit
            if wait_url:
                join_params["wait_url"] = wait_url
            if max_participants != 250:
                join_params["max_participants"] = max_participants
            if record != "do-not-record":
                join_params["record"] = record
            if region:
                join_params["region"] = region
            if trim != "trim-silence":
                join_params["trim"] = trim
            if coach:
                join_params["coach"] = coach
            if status_callback_event:
                join_params["status_callback_event"] = status_callback_event
            if status_callback:
                join_params["status_callback"] = status_callback
            if status_callback_method != "POST":
                join_params["status_callback_method"] = status_callback_method
            if recording_status_callback:
                join_params["recording_status_callback"] = recording_status_callback
            if recording_status_callback_method != "POST":
                join_params["recording_status_callback_method"] = recording_status_callback_method
            if recording_status_callback_event != "completed":
                join_params["recording_status_callback_event"] = recording_status_callback_event
            if result is not None:
                join_params["result"] = result
        
        # Generate SWML document
        swml_doc = {
            "version": "1.0.0",
            "sections": {
                "main": [
                    {"join_conference": join_params}
                ]
            }
        }
        
        # Use execute_swml to add the action
        return self.execute_swml(swml_doc)

    def tap(self, uri: str, control_id: Optional[str] = None, direction: str = "both",
           codec: str = "PCMU", rtp_ptime: int = 20, 
           status_url: Optional[str] = None) -> 'SwaigFunctionResult':
        """
        Start background call tap using SWML.
        
        This is a virtual helper that generates SWML to start background call tapping.
        Media is streamed over Websocket or RTP to customer controlled URI.
        
        Args:
            uri: Destination of tap media stream (required)
                 Formats: rtp://IP:port, ws://example.com, or wss://example.com
            control_id: Identifier for this tap to use with stop_tap (optional)
                        Default is generated and stored in tap_control_id variable
            direction: Direction of audio to tap (default: "both")
                      "speak" = what party says
                      "hear" = what party hears  
                      "both" = what party hears and says
            codec: Codec for tap media stream - "PCMU" or "PCMA" (default: "PCMU")
            rtp_ptime: Packetization time in milliseconds for RTP (default: 20)
            status_url: URL for status change requests (optional)
            
        Returns:
            self for method chaining
            
        Raises:
            ValueError: If direction or codec values are invalid
        """
        # Validate direction parameter
        valid_directions = ["speak", "hear", "both"]
        if direction not in valid_directions:
            raise ValueError(f"direction must be one of {valid_directions}")
        
        # Validate codec parameter
        valid_codecs = ["PCMU", "PCMA"]
        if codec not in valid_codecs:
            raise ValueError(f"codec must be one of {valid_codecs}")
        
        # Validate rtp_ptime
        if rtp_ptime <= 0:
            raise ValueError("rtp_ptime must be a positive integer")
        
        # Build the tap parameters
        tap_params = {"uri": uri}
        
        # Add optional parameters if they differ from defaults
        if control_id:
            tap_params["control_id"] = control_id
        if direction != "both":
            tap_params["direction"] = direction
        if codec != "PCMU":
            tap_params["codec"] = codec
        if rtp_ptime != 20:
            tap_params["rtp_ptime"] = rtp_ptime
        if status_url:
            tap_params["status_url"] = status_url
        
        # Generate SWML document
        swml_doc = {
            "version": "1.0.0",
            "sections": {
                "main": [
                    {"tap": tap_params}
                ]
            }
        }
        
        # Use execute_swml to add the action
        return self.execute_swml(swml_doc)

    def stop_tap(self, control_id: Optional[str] = None) -> 'SwaigFunctionResult':
        """
        Stop an active tap stream using SWML.
        
        This is a virtual helper that generates SWML to stop a tap stream
        that was started with tap().
        
        Args:
            control_id: ID of the tap to stop (optional)
                       If not set, the last tap started will be stopped
            
        Returns:
            self for method chaining
        """
        # Build the stop_tap parameters
        if control_id:
            stop_params = {"control_id": control_id}
        else:
            # For simple case with no control_id, use empty object
            stop_params = {}
        
        # Generate SWML document
        swml_doc = {
            "version": "1.0.0",
            "sections": {
                "main": [
                    {"stop_tap": stop_params}
                ]
            }
        }
        
        # Use execute_swml to add the action
        return self.execute_swml(swml_doc)

    @staticmethod
    def create_payment_prompt(for_situation: str, actions: List[Dict[str, str]], 
                             card_type: Optional[str] = None, 
                             error_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a payment prompt structure for use with pay() method.
        
        Args:
            for_situation: Situation to use prompt for (e.g., "payment-card-number")
            actions: List of actions with 'type' and 'phrase' keys
            card_type: Space-separated card types for this prompt
            error_type: Space-separated error types for this prompt
            
        Returns:
            Dictionary representing the prompt structure
        """
        prompt = {
            "for": for_situation,
            "actions": actions
        }
        
        if card_type:
            prompt["card_type"] = card_type
        if error_type:
            prompt["error_type"] = error_type
            
        return prompt

    @staticmethod
    def create_payment_action(action_type: str, phrase: str) -> Dict[str, str]:
        """
        Create a payment action for use in payment prompts.
        
        Args:
            action_type: "Say" for text-to-speech or "Play" for audio file
            phrase: Sentence to say or URL to play
            
        Returns:
            Dictionary representing the action
        """
        return {
            "type": action_type,
            "phrase": phrase
        }

    @staticmethod
    def create_payment_parameter(name: str, value: str) -> Dict[str, str]:
        """
        Create a payment parameter for use with pay() method.
        
        Args:
            name: Parameter name
            value: Parameter value
            
        Returns:
            Dictionary representing the parameter
        """
        return {
            "name": name,
            "value": value
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to the JSON structure expected by SWAIG
        
        The result must have at least one of:
        - 'response': Text to be spoken by the AI
        - 'action': Array of action objects
        
        Optional:
        - 'post_process': Boolean controlling when actions execute
        
        Returns:
            Dictionary in SWAIG function response format
        """
        # Create the result object
        result = {}
        
        # Add response if present
        if self.response:
            result["response"] = self.response
            
        # Add action if present
        if self.action:
            result["action"] = self.action
            
        # Add post_process if enabled and we have actions
        # (post_process only matters when there are actions to execute)
        if self.post_process and self.action:
            result["post_process"] = True
            
        # Ensure we have at least one of response or action
        if not result:
            # Default response if neither is present
            result["response"] = "Action completed."
            
        return result
