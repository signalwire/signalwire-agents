#!/usr/bin/env python3
"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""


"""
Comprehensive Dynamic Agent Configuration Example

This example demonstrates the power of dynamic agent configuration,
showing how to dynamically configure multiple aspects of an agent based on request parameters.

Features demonstrated:
- Dynamic voice and language selection
- Context-aware prompt configuration  
- Tier-based feature settings
- Industry-specific customization
- A/B testing configuration
- Multi-tenant global data

Usage examples:

1. Premium Healthcare Agent:
   curl "http://localhost:3000/dynamic?tier=premium&industry=healthcare&voice=inworld.Sarah&test_group=A"

2. Standard Finance Agent:
   curl "http://localhost:3000/dynamic?tier=standard&industry=finance&voice=inworld.Mark&test_group=B"

3. Enterprise Retail Agent:
   curl "http://localhost:3000/dynamic?tier=enterprise&industry=retail&voice=inworld.Hanna&language=es&locale=mx"

4. Developer Testing:
   curl "http://localhost:3000/dynamic/debug?tier=premium&debug=true"

Query Parameters:
- tier: standard|premium|enterprise (affects features and timeouts)
- industry: healthcare|finance|retail|general (customizes prompts and responses)
- voice: inworld.Mark|inworld.Sarah|inworld.Hanna|inworld.Blake (voice selection)
- language: en|es|fr (language support)  
- locale: us|mx|ca|fr (locale-specific customization)
- test_group: A|B (A/B testing configuration)
- debug: true (enables debug features)
- customer_id: <string> (customer-specific data)
"""

from signalwire_agents import AgentBase

class ComprehensiveDynamicAgent(AgentBase):
    def __init__(self, route="/dynamic"):
        super().__init__(
            name="Comprehensive Dynamic Agent",
            route=route,  # Custom path for this agent
            auto_answer=True,
            record_call=True
        )
        
        # Set up dynamic configuration
        self.set_dynamic_config_callback(self.configure_agent_dynamically)
        
        # Define available voice options by tier
        self.voice_options = {
            "standard": ["inworld.Mark", "inworld.Sarah", "inworld.Blake"],
            "premium": ["inworld.Sarah", "inworld.Hanna", "inworld.Mark"],
            "enterprise": ["inworld.Mark", "inworld.Sarah", "inworld.Hanna", "inworld.Blake"]
        }
        
        # Industry-specific configurations
        self.industry_configs = {
            "healthcare": {
                "compliance_level": "high",
                "privacy_emphasis": True,
                "terminology": "medical",
                "response_style": "professional"
            },
            "finance": {
                "compliance_level": "high", 
                "security_emphasis": True,
                "terminology": "financial",
                "response_style": "formal"
            },
            "retail": {
                "compliance_level": "medium",
                "sales_emphasis": True,
                "terminology": "customer_service",
                "response_style": "friendly"
            },
            "general": {
                "compliance_level": "standard",
                "terminology": "general",
                "response_style": "conversational"
            }
        }

    def configure_agent_dynamically(self, query_params, body_params, headers, agent):
        """
        Dynamic configuration callback that demonstrates comprehensive agent customization
        """
        # Extract key parameters
        tier = query_params.get('tier', 'standard').lower()
        industry = query_params.get('industry', 'general').lower()
        requested_voice = query_params.get('voice', '').lower()
        language = query_params.get('language', 'en').lower()
        locale = query_params.get('locale', 'us').lower()
        test_group = query_params.get('test_group', 'A').upper()
        debug_mode = query_params.get('debug', '').lower() == 'true'
        customer_id = query_params.get('customer_id', '')
        
        # === Voice Configuration ===
        self._configure_voice_and_language(agent, tier, requested_voice, language, locale)
        
        # === Tier-based Parameters ===
        self._configure_tier_parameters(agent, tier, test_group)
        
        # === Industry-specific Prompts ===
        self._configure_industry_prompts(agent, industry, tier)
        
        # === Global Data Setup ===
        self._configure_global_data(agent, tier, industry, customer_id, test_group, debug_mode)
        
        # === Debug Features ===
        if debug_mode:
            self._configure_debug_features(agent, query_params)
            
        # === A/B Testing Configuration ===
        self._configure_ab_testing(agent, test_group, tier)

    def _configure_voice_and_language(self, agent, tier, requested_voice, language, locale):
        """Configure voice and language settings"""
        # Determine available voices for tier
        available_voices = self.voice_options.get(tier, self.voice_options["standard"])
        
        # Select voice (use requested if available for tier, otherwise default)
        if requested_voice and requested_voice in available_voices:
            voice = requested_voice
        else:
            voice = available_voices[0]  # Default to first available
        
        if language == "en":
            if locale == "us":
                agent.add_language("English US", "en-US", voice)
            elif locale == "ca":
                agent.add_language("English CA", "en-CA", voice)
            else:
                agent.add_language("English", "en-US", voice)
        elif language == "es":
            # Spanish configuration
            if locale == "mx":
                agent.add_language("Spanish MX", "es-MX", "inworld.Sarah")
            else:
                agent.add_language("Spanish", "es-ES", "inworld.Sarah")
        elif language == "fr":
            if locale == "ca":
                agent.add_language("French CA", "fr-CA", "inworld.Hanna")
            else:
                agent.add_language("French", "fr-FR", "inworld.Hanna")
        else:
            # Default to English
            agent.add_language("English", "en-US", voice)

    def _configure_tier_parameters(self, agent, tier, test_group):
        """Configure parameters based on service tier"""
        params = {}
        
        if tier == "enterprise":
            params.update({
                "end_of_speech_timeout": 800,
                "attention_timeout": 25000,
                "background_file_volume": -35,
                "background_file_loops": -1,
                "digit_timeout": 8000,
                "energy_level": 150
            })
        elif tier == "premium":
            params.update({
                "end_of_speech_timeout": 600,
                "attention_timeout": 20000,
                "background_file_volume": -30,
                "background_file_loops": 5,
                "digit_timeout": 6000,
                "energy_level": 120
            })
        else:  # standard
            params.update({
                "end_of_speech_timeout": 400,
                "attention_timeout": 15000,
                "background_file_volume": -20,
                "background_file_loops": 3,
                "digit_timeout": 4000,
                "energy_level": 100
            })
            
        # A/B test parameter variation
        if test_group == "B":
            params["end_of_speech_timeout"] = int(params["end_of_speech_timeout"] * 1.2)
            
        # Add ai_model to all configurations
        params["ai_model"] = "gpt-4.1-nano"
            
        agent.set_params(params)

    def _configure_industry_prompts(self, agent, industry, tier):
        """Configure industry-specific prompts"""
        config = self.industry_configs.get(industry, self.industry_configs["general"])
        
        # Base introduction
        agent.prompt_add_section(
            "Role and Purpose",
            f"You are a professional AI assistant specialized in {industry} services. "
            f"Your role is to provide helpful, accurate information while maintaining "
            f"{config['response_style']} communication standards."
        )
        
        # Industry-specific guidelines
        if industry == "healthcare":
            agent.prompt_add_section(
                "Healthcare Guidelines",
                "Follow HIPAA compliance standards. Never provide medical diagnoses or treatment advice.",
                bullets=[
                    "Protect patient privacy at all times",
                    "Direct medical questions to qualified healthcare providers", 
                    "Use appropriate medical terminology when helpful",
                    "Maintain professional bedside manner"
                ]
            )
        elif industry == "finance":
            agent.prompt_add_section(
                "Financial Guidelines", 
                "Adhere to financial industry regulations and maintain strict confidentiality.",
                bullets=[
                    "Never provide specific investment advice",
                    "Protect sensitive financial information",
                    "Use precise financial terminology",
                    "Refer complex matters to qualified advisors"
                ]
            )
        elif industry == "retail":
            agent.prompt_add_section(
                "Customer Service Excellence",
                "Focus on customer satisfaction and sales support.",
                bullets=[
                    "Maintain friendly, helpful demeanor",
                    "Understand product features and benefits",
                    "Handle complaints with empathy",
                    "Look for opportunities to enhance customer experience"
                ]
            )
        
        # Tier-specific capabilities
        if tier in ["premium", "enterprise"]:
            agent.prompt_add_section(
                "Enhanced Capabilities",
                f"As a {tier} service, you have access to advanced features:",
                bullets=[
                    "Extended conversation memory",
                    "Priority processing and faster responses", 
                    "Access to specialized knowledge bases",
                    "Advanced personalization options"
                ]
            )

    def _configure_global_data(self, agent, tier, industry, customer_id, test_group, debug_mode):
        """Configure global data for the session"""
        global_data = {
            "service_tier": tier,
            "industry_focus": industry,
            "test_group": test_group,
            "session_start": "2024-01-01T00:00:00Z",  # Would be actual timestamp
            "features_enabled": self._get_enabled_features(tier),
            "compliance_level": self.industry_configs[industry]["compliance_level"]
        }
        
        if customer_id:
            global_data["customer_id"] = customer_id
            global_data["customer_tier"] = tier
            
        if debug_mode:
            global_data["debug_mode"] = True
            global_data["debug_info"] = {
                "voice_options": self.voice_options[tier],
                "available_features": self._get_enabled_features(tier)
            }
            
        agent.set_global_data(global_data)

    def _configure_debug_features(self, agent, query_params):
        """Configure debug-specific features"""
        agent.prompt_add_section(
            "Debug Mode",
            "Debug mode is enabled. Provide additional context and reasoning in responses.",
            bullets=[
                "Show decision-making process when appropriate",
                "Include relevant global data references",
                "Explain feature availability based on tier",
                "Provide timing and performance insights"
            ]
        )
        
        # Add debug-specific speech recognition hints
        agent.add_hints([
            "debug",
            "verbose",
            "reasoning",
            "capabilities",
            "tier"
        ])

    def _configure_ab_testing(self, agent, test_group, tier):
        """Configure A/B testing variations"""
        if test_group == "A":
            # Control group - add speech recognition hints for standard terms
            agent.add_hint("standard")
        else:
            # Test group B - add speech recognition hints for enhanced features
            agent.add_hints(["enhanced", "personalized", "proactive"])
            agent.prompt_add_section(
                "Enhanced Interaction Style",
                "You are using an enhanced conversation style for this session:",
                bullets=[
                    "Ask clarifying questions more frequently",
                    "Provide more detailed explanations",
                    "Offer proactive suggestions when appropriate",
                    "Use more conversational, engaging language"
                ]
            )

    def _get_enabled_features(self, tier):
        """Return list of features enabled for the given tier"""
        features = ["basic_conversation", "function_calling"]
        
        if tier in ["premium", "enterprise"]:
            features.extend([
                "extended_memory",
                "priority_processing", 
                "advanced_analytics"
            ])
            
        if tier == "enterprise":
            features.extend([
                "custom_integration",
                "dedicated_support",
                "advanced_security"
            ])
            
        return features


# Example usage and testing
def main():
    agent = ComprehensiveDynamicAgent()
    
    print("\nStarting agent server...")
    print("Note: Works in any deployment mode (server/CGI/Lambda)")
    agent.run()

if __name__ == "__main__":
    main() 