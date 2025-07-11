--
-- PostgreSQL database dump
--

-- Dumped from database version 17.2 (Debian 17.2-1.pgdg120+1)
-- Dumped by pg_dump version 17.2 (Debian 17.2-1.pgdg120+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: update_updated_at_column(); Type: FUNCTION; Schema: public; Owner: postgres
--

CREATE FUNCTION public.update_updated_at_column() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
BEGIN
    NEW.updated_at = NOW();
RETURN NEW;
END;
$$;


ALTER FUNCTION public.update_updated_at_column() OWNER TO postgres;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: fulfill_order; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.fulfill_order (
                                      order_id character varying(100) NOT NULL,
                                      fulfill_status character varying(100) NOT NULL,
                                      maker_transfer_tx character varying(100),
                                      taker_transfer_tx character varying(100),
                                      created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
                                      updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP NOT NULL,
                                      err_msg text,
                                      input_volume_usd character varying(32),
                                      output_volume_usd character varying(32),
                                      input_gas_usd character varying(32),
                                      output_gas_usd character varying(32),
                                      total_gas_usd character varying(32)
);


ALTER TABLE public.fulfill_order OWNER TO postgres;

--
-- Name: maker_order; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.maker_order (
                                    order_id character varying(100) NOT NULL,
                                    cobo_id character varying(100),
                                    wallet_id character varying(100) NOT NULL,
                                    source_chain character varying(100) NOT NULL,
                                    target_chain character varying(100) NOT NULL,
                                    i_token character varying(100) NOT NULL,
                                    i_amount character varying(100) NOT NULL,
                                    o_amount character varying(100) NOT NULL,
                                    maker_tx_hash character varying(100) NOT NULL,
                                    status character varying(100) NOT NULL,
                                    fulfill_status character varying(100) NOT NULL,
                                    slippage double precision DEFAULT 0 NOT NULL,
                                    o_token character varying(100),
                                    created_at timestamp without time zone DEFAULT now() NOT NULL,
                                    updated_at timestamp without time zone DEFAULT now() NOT NULL,
                                    sig character varying(300) DEFAULT ''::character varying
);


ALTER TABLE public.maker_order OWNER TO postgres;

--
-- Name: mpc_addr; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.mpc_addr (
                                 wallet_id character varying(100) NOT NULL,
                                 chain_id character varying(100) NOT NULL,
                                 address character varying(100) NOT NULL,
                                 encoding character varying(100)
);


ALTER TABLE public.mpc_addr OWNER TO postgres;

--
-- Name: mpc_wallet; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.mpc_wallet (
                                   wallet_id character varying(100) NOT NULL,
                                   src_addr character varying(100) NOT NULL,
                                   wallet_name character varying(100) NOT NULL
);


ALTER TABLE public.mpc_wallet OWNER TO postgres;

--
-- Name: taker_order; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.taker_order (
                                    order_id character varying(100) NOT NULL,
                                    cobo_id character varying(100),
                                    wallet_id character varying(100) NOT NULL,
                                    token character varying(100) NOT NULL,
                                    taker_tx_hash character varying(100) NOT NULL,
                                    amount character varying(100) NOT NULL,
                                    status character varying(100) NOT NULL,
                                    created_at timestamp without time zone DEFAULT now() NOT NULL,
                                    updated_at timestamp without time zone DEFAULT now() NOT NULL
);


ALTER TABLE public.taker_order OWNER TO postgres;

--
-- Name: maker_order maker_order_cobo_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.maker_order
    ADD CONSTRAINT maker_order_cobo_id_key UNIQUE (cobo_id);


--
-- Name: maker_order maker_order_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.maker_order
    ADD CONSTRAINT maker_order_pkey PRIMARY KEY (order_id);


--
-- Name: mpc_addr mpc_addr_wallet_id_chain_id_address_encoding_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.mpc_addr
    ADD CONSTRAINT mpc_addr_wallet_id_chain_id_address_encoding_key UNIQUE (wallet_id, chain_id, address, encoding);


--
-- Name: mpc_wallet mpc_wallet_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.mpc_wallet
    ADD CONSTRAINT mpc_wallet_pkey PRIMARY KEY (wallet_id);


--
-- Name: taker_order taker_order_cobo_id_key; Type: CONSTRAINT; Schema: public; Owner: postgres
--

-- ALTER TABLE ONLY public.taker_order
--     ADD CONSTRAINT taker_order_cobo_id_key UNIQUE (cobo_id);


--
-- Name: fulfill_order update_fulfill_order_updated_at; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER update_fulfill_order_updated_at BEFORE UPDATE ON public.fulfill_order FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: maker_order update_maker_order_updated_at; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER update_maker_order_updated_at BEFORE UPDATE ON public.maker_order FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: taker_order update_taker_order_updated_at; Type: TRIGGER; Schema: public; Owner: postgres
--

CREATE TRIGGER update_taker_order_updated_at BEFORE UPDATE ON public.taker_order FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();


--
-- Name: mpc_addr mpc_addr_wallet_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.mpc_addr
    ADD CONSTRAINT mpc_addr_wallet_id_fkey FOREIGN KEY (wallet_id) REFERENCES public.mpc_wallet(wallet_id);


--
-- Name: taker_order taker_order_order_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.taker_order
    ADD CONSTRAINT taker_order_order_id_fkey FOREIGN KEY (order_id) REFERENCES public.maker_order(order_id);


CREATE TABLE public.solvers (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    api_endpoint TEXT NOT NULL,
    supported_network TEXT[] NOT NULL,
    supported_asset JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO public.solvers (name, email, api_endpoint, supported_network, supported_asset)
VALUES (
           'fluxlayer',
           'fluxlayer@gamil.com',
           'http://localhost:15886/get_best_rfq',
           ARRAY['BTC', '"SIGNET_BTC"', 'SOL'],
           '{
             "BTC": {"BTC": true},
             "SIGNET_BTC": {"BTC": true},
             "SOL": {"SOL": true, "USDC": true}
           }'::jsonb
       );

--
-- PostgreSQL database dump complete
--
